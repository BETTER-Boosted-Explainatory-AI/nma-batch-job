class CycleFound(Exception):
    """Raised when a cycle is detected in an acyclic-expected structure."""

def assert_acyclic(tree):
    """
    Depth-first walk without recursion.
    Raises CycleFound if the same logical node id is visited twice.
    """
    seen  = set()
    stack = [tree]

    while stack:
        node = stack.pop()
        node_id = node.get("id")          # every node produced by
                                          # _build_tree_hierarchy has one
        if node_id in seen:
            raise CycleFound(f"cycle at logical id {node_id}")
        seen.add(node_id)

        # push children (if any) onto our own stack
        stack.extend(node.get("children", []))
