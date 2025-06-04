from nltk.corpus import wordnet as wn
import os
import boto3
import re
from NMA.utilss.s3_utils import get_datasets_s3_client
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
## NEW VERSION - 3/6/25


def convert_folder_names_to_readable_labels(dataset_path):
    """
    Convert folder names in S3 to readable labels using WordNet.
    
    Parameters:
    - dataset_path: S3 path in the format 's3://bucket/prefix' or just 'prefix'
                   (will use S3_USERS_BUCKET_NAME env var)
    
    Returns:
    - readable_labels: List of readable class labels
    - folder_to_label: Dictionary mapping folder names to readable labels
    """
    s3_client = get_datasets_s3_client()
    
    if dataset_path.startswith('s3://'):
        parts = dataset_path.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''
    else:
        bucket = os.getenv("S3_DATASETS_BUCKET_NAME")
        if not bucket:
            raise ValueError("S3_DATASETS_BUCKET_NAME environment variable is required when not using full s3:// path")
        prefix = dataset_path
    
    if prefix and not prefix.endswith('/'):
        prefix = prefix + '/'
    
    try:
        s3_client.head_object(Bucket=bucket, Key=prefix)
    except:
        # Try listing objects to see if prefix exists as a directory
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/', MaxKeys=1)
        if 'CommonPrefixes' not in response and 'Contents' not in response:
            raise ValueError(f"Dataset path does not exist in S3: {bucket}/{prefix}")
    
    class_names = []
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/')
    
    for page in pages:
        if 'CommonPrefixes' in page:
            for common_prefix in page['CommonPrefixes']:
                # Extract the directory name from the prefix
                dir_name = common_prefix['Prefix'].rstrip('/')
                dir_name = dir_name.split('/')[-1]  # Get the last part of the path
                
                # Check if it follows the pattern 'n' followed by digits
                if dir_name.startswith('n') and dir_name[1:].isdigit():
                    class_names.append(dir_name)
    
    # class_names.sort()
    
    if not class_names:
        raise ValueError(f"No subdirectories found in S3 path {bucket}/{prefix}")
    
    # Define special case mapping for disambiguation
    special_case_mapping = {
        "n02012849": "crane_bird",       # Crane bird
        "n03126707": "crane_machine",    # Crane machine
        "n03710637": "maillot",          # Maillot (swimsuit)
        "n03710721": "tank_suit"         # Tank suit (different type of swimsuit)
    }
    
    # Create mapping from folder names to readable labels
    folder_to_label = {}
    for folder_name in class_names:
        if folder_name in special_case_mapping:
            # Use special case mapping for known ambiguous folders
            readable_label = special_case_mapping[folder_name]
        elif folder_name.startswith('n'):
            # Use WordNet utility for standard cases
            readable_label = WordnetUtils.convert_folder_name_to_label(folder_name)
        else:
            readable_label = folder_name
            
        folder_to_label[folder_name] = readable_label
    
    # Convert original folder names to readable labels while preserving order
    readable_labels = [folder_to_label[folder_name] for folder_name in class_names]
    
    print(f"Sample conversions:")
    for i in range(min(5, len(class_names))):
        print(f"{class_names[i]} → {readable_labels[i]}")
    
    # Print specific indices for debugging
    if len(readable_labels) > 638:
        print(f"Label at index 638: {readable_labels[638]}")
    if len(readable_labels) > 639:
        print(f"Label at index 639: {readable_labels[639]}")
        
    return readable_labels, folder_to_label
    
    
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
    """Extract all leaf node names from a cluster hierarchy."""
    if "children" not in node:
        # Only return actual object names, not cluster names
        if "Cluster" not in node["name"]:
            return [node["name"]]
        return []
    
    names = []
    for child in node["children"]:
        names.extend(get_all_leaf_names(child))
    return names


def process_hierarchy(hierarchy_data, debug=False):
    """Process the entire hierarchy, renaming clusters while preserving structure."""
    if debug:
        logger.info("Starting hierarchy processing with improved naming")
    return _rename_clusters(hierarchy_data, debug=debug)



# ---------------------------------------------------
# Helper: fetch up to `MAX_SENSES` synsets for a phrase
# ---------------------------------------------------
def _get_top_synsets(
    phrase: str,
    pos=wn.NOUN,
    max_senses: int = 5
) -> list[wn.synset]:
    """
    Return up to `max_senses` synsets for `phrase`.
    - Replaces spaces/underscores so WordNet can match “pickup truck” or “aquarium_fish”.
    - WordNet already orders synsets by frequency, so we take only the first few.
    """
    lemma = phrase.strip().lower().replace(" ", "_")
    syns = wn.synsets(lemma, pos=pos)
    return syns[:max_senses] if syns else []


# ---------------------------------------------------
# Core: compute the single best hypernym for a set of words
# ---------------------------------------------------
def _find_best_common_hypernym(
    leaves: list[str],
    max_senses_per_word: int = 5,
    banned_lemmas: set[str] = None,
    debug: bool = False
) -> str | None:
    """
    1. For each leaf in `leaves`, fetch up to `max_senses_per_word` synsets.
    2. For EVERY pair of leaves (w1, w2), for EVERY combination of synset ∈ synsets(w1) × synsets(w2),
       call syn1.lowest_common_hypernyms(syn2) → yields a list of shared hypernyms.
       Tally them in `lch_counter`.
    3. Sort the candidates by (frequency, min_depth) so we pick the most-specific, most-common ancestor.
    4. Filter out overly generic lemmas (like “entity”, “object”) unless NOTHING else remains.
    5. Return the best lemma_name (underscore → space, capitalized).
    """
    if banned_lemmas is None:
        banned_lemmas = {"entity", "object", "physical_entity", "thing", "Object", "Whole", "Whole", "Physical_entity", "Thing", "Entity", "Artifact"}

    if debug:
        logger.info(f"[find_best_common_hypernym] leaves: {leaves}")

    # 1. Map each leaf → up to `max_senses_per_word` synsets
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

    # If fewer than 2 words have ANY synsets, we cannot get a meaningful common hypernym
    if len(word_to_synsets) < 2:
        if debug:
            logger.info("Less than 2 leaves had WordNet synsets → returning None")
        return None

    # 2. For each pair of distinct leaves w1, w2, do ALL combinations of synset₁ × synset₂
    #    and tally lowest_common_hypernyms
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

    # 3. Sort candidates by (frequency, min_depth) descending
    candidates = sorted(
        lch_counter.items(),
        key=lambda item: (item[1], item[0].min_depth()),
        reverse=True
    )

    if debug:
        top_info = [(syn.name(), freq, syn.min_depth()) for syn, freq in candidates[:5]]
        logger.info(f"Top hypernym candidates: {top_info}")

    # 4. Filter out generic lemma_names unless NOTHING else remains
    filtered: list[tuple[wn.synset, int]] = []
    for syn, freq in candidates:
        lemma = syn.name().split(".")[0].lower()
        if lemma in banned_lemmas:
            continue
        filtered.append((syn, freq))

    # If every candidate was filtered out, allow the first generic anyway
    if not filtered:
        filtered = candidates

    best_synset, best_freq = filtered[0]
    best_label = (best_synset.name().split(".")[0].replace(" ", "_")).lower()
    if debug:
        logger.info(f"Selected hypernym: '{best_label}'  (freq={best_freq}, depth={best_synset.min_depth()})")
    return best_label


# ---------------------------------------------------
# Public version: branching on single vs. multiple leaves
# ---------------------------------------------------
def find_common_hypernyms(
    words: list[str],
    abstraction_level: int = 0,
    debug: bool = False
) -> str | None:
    """
    Improved drop-in replacement for your old `find_common_hypernyms`.
    1. Normalize each word (underscores ↔ spaces, lowercase) and filter out anything containing "Cluster".
    2. If there’s exactly one valid leaf, pick its first hypernym (one level up) unless it’s “entity”.
    3. If there are 2+ leaves, call _find_best_common_hypernym on them.
    """
    if debug:
        logger.info(f"[find_common_hypernyms] input words: {words}  abstraction_level={abstraction_level}")

    clean_leaves = [
        w.strip().lower().replace(" ", "_")
        for w in words
        if w and "cluster" not in w.lower()
    ]

    if debug:
        logger.info(f"Cleaned leaves: {clean_leaves}")

    # If nothing remains, bail out
    if not clean_leaves:
        if debug:
            logger.info("No valid leaves → returning None")
        return None

    # Single-word case: pick its immediate hypernym (second-to-bottom in the hypernym path)
    if len(clean_leaves) == 1:
        word = clean_leaves[0]
        synsets = _get_top_synsets(word, wn.NOUN, max_senses=10)
        if not synsets:
            return None

        # Choose the first sense’s longest hypernym path, then take one level up from leaf sense.
        paths = synsets[0].hypernym_paths()  # list of lists
        if not paths:
            return None

        longest_path = max(paths, key=lambda p: len(p))
        # If path has at least 2 nodes, candidate = one level above the leaf sense
        if len(longest_path) >= 2:
            candidate = longest_path[-2]
            name = (candidate.name().split(".")[0].replace(" ", "_")).lower()
            if name.lower() not in {word, "entity"}:
                if debug:
                    logger.info(f"Single-leaf hypernym = '{name}'")
                return name
        return None

    # 2+ leaves: use pairwise LCA approach
    return _find_best_common_hypernym(clean_leaves, max_senses_per_word=5, debug=debug)

def _rename_clusters(node, depth=0, used_names=None, all_leaf_names=None, debug=False):
    if used_names is None:
        used_names = set()
    if all_leaf_names is None:
        all_leaves = get_all_leaf_names(node)
        all_leaf_names = {n.lower() for n in all_leaves}
        if debug:
            logger.info(f"All leaf names: {list(all_leaf_names)[:10]}…")

    # Recurse children first
    if "children" in node:
        for i, child in enumerate(node["children"]):
            node["children"][i] = _rename_clusters(
                child, depth + 1, used_names, all_leaf_names, debug=debug
            )

    # If this node’s name still starts with “Cluster”, attempt to rename
    if "Cluster" in node["name"]:
        if debug:
            logger.info(f"Processing cluster: {node['name']}")

        leaf_names = get_all_leaf_names(node)
        if not leaf_names:
            if debug:
                logger.info("  → no leaves under this cluster, keep name")
            return node

        # Call the new LCA-based finder (no more abstraction loops)
        candidate = find_common_hypernyms(leaf_names, debug=debug)

        if candidate:
            # Ensure it doesn’t conflict with actual leaf names or prev used names
            base = candidate
            unique = base
            idx = 1
            while unique.lower() in all_leaf_names or unique.lower() in {n.lower() for n in used_names}:
                idx += 1
                unique = f"{base}_{idx}"
            node["name"] = unique
            used_names.add(unique)
            if debug:
                logger.info(f"  → renamed cluster to '{unique}'")
        else:
            if debug:
                logger.info("  → no good hypernym found, keep 'Cluster …'")

    return node
