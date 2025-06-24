from nltk.corpus import wordnet as wn
from typing import Optional
import re
import logging
from collections import deque
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from itertools import combinations, product
from collections import Counter

def folder_name_to_number(folder_name):
    synsets = wn.synsets(folder_name)
    if synsets:
        offset = synsets[0].offset()        
        folder_number = 'n{:08d}'.format(offset)
        return folder_number
    
def common_group(groups):
    common_hypernyms = []
    hierarchy = {}
    for group in groups:
        hierarchy[group] = []
        synsets = wn.synsets(group)
        if synsets:
            hypernyms = synsets[0].hypernym_paths()
            for path in hypernyms:
                hierarchy[group].extend([node.name().split('.')[0] for node in path])
                
    if len(hierarchy) == 1:
        common_hypernyms = list(hierarchy.values())[0]
    else:
        for hypernym in hierarchy[groups.pop()]:
            if all(hypernym in hypernyms for hypernyms in hierarchy.values()):
                common_hypernyms.append(hypernym)
    return common_hypernyms[::-1]


def get_all_leaf_names(node):
    if "children" not in node:
        if "Cluster" not in node["name"]:
            return [node["name"]]
        return []
    names = []
    for child in node["children"]:
        names.extend(get_all_leaf_names(child))
    return names
    

def process_hierarchy(hierarchy_data, debug=False):
    if debug:
        logger.info("Starting hierarchy processing with improved naming")
    return _rename_clusters(hierarchy_data, debug=debug)


def _get_top_synsets(phrase: str, pos=wn.NOUN, max_senses: int = 15) -> list[wn.synset]:
    lemma = phrase.strip().lower().replace(" ", "_")
    syns = wn.synsets(lemma, pos=pos)
    return syns[:max_senses] if syns else []


def _find_best_common_hypernym(
    leaves: list[str],
    max_senses_per_word: int = 5,
    banned_lemmas: set[str] = None,
    debug: bool = False
) -> Optional[str]:
    
    if banned_lemmas is None:
        banned_lemmas = {"entity", "object", "physical_entity", "thing", "Object", "Whole", "Whole", "Physical_entity", "Thing", "Entity", "Artifact"}
    if debug:
        logger.info(f"[find_best_common_hypernym] leaves: {leaves}")
    word_to_synsets: dict[str, list[wn.synset]] = {}
    for w in leaves:
        syns = _get_top_synsets(w, wn.NOUN, max_senses_per_word)
        if syns:
            word_to_synsets[w] = syns
            if debug:
                logger.info(f"Word '{w}' → {len(syns)} synsets: {[s.name() for s in syns]}")
        else:
            if debug:
                logger.info(f"No synsets found for '{w}'")

    if len(word_to_synsets) < 2:
        if debug:
            logger.info("Less than 2 leaves had WordNet synsets → returning None")
        return None

    lch_counter: Counter[wn.synset] = Counter()
    words_list = list(word_to_synsets.keys())

    for w1, w2 in combinations(words_list, 2):
        syns1 = word_to_synsets[w1]
        syns2 = word_to_synsets[w2]
        if debug:
            logger.info(f"→ Pair: {w1} ({len(syns1)} senses)  ×  {w2} ({len(syns2)} senses)")

        for s1, s2 in product(syns1, syns2):
            try:
                common = s1.lowest_common_hypernyms(s2)
            except Exception as e:
                if debug:
                    logger.warning(f"Error computing LCH({s1.name()}, {s2.name()}): {e}")
                continue
            for hyp in common:
                lch_counter[hyp] += 1

    if not lch_counter:
        if debug:
            logger.info("No lowest_common_hypernyms found across any synset pairs → returning None")
        return None

    candidates = sorted(
        lch_counter.items(),
        key=lambda item: (item[1], item[0].min_depth()),
        reverse=True
    )

    if debug:
        top_info = [(syn.name(), freq, syn.min_depth()) for syn, freq in candidates[:5]]
        logger.info(f"Top hypernym candidates: {top_info}")

    filtered: list[tuple[wn.synset, int]] = []
    for syn, freq in candidates:
        lemma = syn.name().split(".")[0].lower()
        if lemma in banned_lemmas:
            continue
        filtered.append((syn, freq))

    if not filtered:
        filtered = candidates

    best_synset, best_freq = filtered[0]
    best_label = (best_synset.name().split(".")[0].replace(" ", "_")).lower()
    if debug:
        logger.info(f"Selected hypernym: '{best_label}'  (freq={best_freq}, depth={best_synset.min_depth()})")
    return best_label

def find_common_hypernyms(
    words: list[str],
    abstraction_level: int = 0,
    debug: bool = False
) -> Optional[str]:
  
    if debug:
        logger.info(f"[find_common_hypernyms] input words: {words}  abstraction_level={abstraction_level}")

    clean_leaves = [
        re.sub(r'_\d+$', '', w.strip().lower().replace(" ", "_"))
        for w in words
        if w and "cluster" not in w.lower()
    ]
    print("clean_leaves",clean_leaves)


    if debug:
        logger.info(f"Cleaned leaves: {clean_leaves}")

    if not clean_leaves:
        if debug:
            logger.info("No valid leaves → returning None")
        return None

    if len(clean_leaves) == 1:
        word = clean_leaves[0]
        synsets = _get_top_synsets(word, wn.NOUN, max_senses=10)
        if not synsets:
            return None

        paths = synsets[0].hypernym_paths()  # list of lists
        if not paths:
            return None

        longest_path = max(paths, key=lambda p: len(p))
        if len(longest_path) >= 2:
            candidate = longest_path[-2]
            name = (candidate.name().split(".")[0].replace(" ", "_")).lower()
            if name.lower() not in {word, "entity"}:
                if debug:
                    logger.info(f"Single-leaf hypernym = '{name}'")
                return name
        return None
    return _find_best_common_hypernym(clean_leaves, max_senses_per_word=5, debug=debug)


def _rename_clusters(tree, debug=False):
    used_names = set()
    all_leaf_names = {leaf.lower() for leaf in get_all_leaf_names(tree)}
    queue = deque()
    queue.append(tree)
    postprocess_nodes = []

    while queue:
        node = queue.popleft()
        if "children" in node:
            queue.extend(node["children"])
            postprocess_nodes.append(node) 

    for node in reversed(postprocess_nodes):
        if "Cluster" not in node["name"]:
            continue 

        child_names = [child["name"] for child in node["children"] if "name" in child]
        candidate = find_common_hypernyms(child_names, debug=debug)

        if candidate:
            base = candidate
            unique = base
            idx = 1
            while unique.lower() in all_leaf_names or unique.lower() in {n.lower() for n in used_names}:
                idx += 1
                unique = f"{base}_{idx}"
            node["name"] = unique
            used_names.add(unique)
            if debug:
                logger.info(f"Renamed cluster {node['id']} to '{unique}'")
        else:
            if debug:
                logger.info(f"No hypernym found for cluster {node['id']} → keeping original name")
    return tree