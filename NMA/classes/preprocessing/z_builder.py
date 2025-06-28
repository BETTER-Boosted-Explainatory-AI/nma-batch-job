import numpy as np


class ZBuilder:
    def create_z_matrix_from_tree(self, clustering_builder, labels):
        """
        Create a Z matrix directly from the final tree.
        Modified to handle duplicate labels correctly.
        """
        # First get unique labels while preserving order where possible
        unique_labels = []
        seen_labels = set()

        for label in labels:
            if label not in seen_labels:
                unique_labels.append(label)
                seen_labels.add(label)

        print(f"Total labels: {len(labels)}, Unique labels: {len(unique_labels)}")

        # Create mapping from unique labels to indices
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}

        n = len(unique_labels)
        # Initialize Z matrix with the correct dimensions for unique labels
        z_matrix = np.zeros((n - 1, 4), dtype=np.float64)

        # Get the root of the tree
        if not clustering_builder.forest:
            print("ERROR: No tree found in the forest!")
            return z_matrix, unique_labels

        root = clustering_builder.forest[0]

        # Keep track of processed nodes and their Z indices
        processed = {}  # node_id -> z_idx or original_idx
        next_z_idx = n  # Start internal node indices at n
        row_idx = 0  # Current row in Z matrix

        # Function to process a node and get its index
        def process_node(node):
            nonlocal row_idx, next_z_idx

            # If we've already processed this node, return its index
            if node.node_id in processed:
                return processed[node.node_id]

            # If this is a leaf node, map it to its index in the unique labels
            if len(node) == 0:  # No children
                if hasattr(node, "node_name") and node.node_name in label_to_idx:
                    idx = label_to_idx[node.node_name]
                    processed[node.node_id] = idx
                    return idx
                else:
                    # If it's not in our label set, skip it
                    return None

            # Process children first
            left_idx = None
            right_idx = None

            if len(node) >= 1:
                left_idx = process_node(node[0])
            if len(node) >= 2:
                right_idx = process_node(node[1])

            # If we don't have both children, we can't create a merge
            if left_idx is None or right_idx is None:
                return None

            # Make sure left_idx < right_idx (scipy convention)
            if left_idx > right_idx:
                left_idx, right_idx = right_idx, left_idx

            # Calculate count of leaves
            left_count = 1 if left_idx < n else z_matrix[left_idx - n][3]
            right_count = 1 if right_idx < n else z_matrix[right_idx - n][3]

            # Add the merge to the Z matrix
            if row_idx < n - 1:  # Ensure we don't exceed matrix dimensions
                z_matrix[row_idx] = [
                    left_idx,
                    right_idx,
                    node.weight,
                    left_count + right_count,
                ]

                # Record this node's index
                this_idx = next_z_idx
                processed[node.node_id] = this_idx

                # Update indices
                next_z_idx += 1
                row_idx += 1

                return this_idx

            return None

        # Process the tree starting from the root
        process_node(root)

        # Check if we filled all rows
        if row_idx < n - 1:
            print(f"WARNING: Only filled {row_idx} of {n-1} rows in Z matrix!")
            # Trim the matrix to the actual number of rows we filled
            z_matrix = z_matrix[:row_idx]

        # Additional validation to ensure Z matrix is valid
        for i in range(z_matrix.shape[0]):
            # Ensure cluster sizes are correct
            left_idx = int(z_matrix[i, 0])
            right_idx = int(z_matrix[i, 1])

            left_size = 1 if left_idx < n else z_matrix[left_idx - n, 3]
            right_size = 1 if right_idx < n else z_matrix[right_idx - n, 3]

            # Fix the size if it's incorrect
            expected_size = left_size + right_size
            if z_matrix[i, 3] != expected_size:
                print(
                    f"Fixing cluster size at row {i}: {z_matrix[i, 3]} → {expected_size}"
                )
                z_matrix[i, 3] = expected_size

            # Ensure cluster size doesn't exceed the number of rows + 1
            max_size = z_matrix.shape[0] + 1
            if z_matrix[i, 3] > max_size:
                print(
                    f"Capping excessive cluster size at row {i}: {z_matrix[i, 3]} → {max_size}"
                )
                z_matrix[i, 3] = max_size

        return np.asarray(z_matrix, dtype=np.float64)
