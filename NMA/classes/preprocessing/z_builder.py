import numpy as np

class ZBuilder:
    def create_z_matrix_from_tree(clustering_builder, labels):
        """
        Create a Z matrix directly from the final tree.
        Modified to handle duplicate labels correctly.
        """
        unique_labels = []
        seen_labels = set()

        for label in labels:
            if label not in seen_labels:
                unique_labels.append(label)
                seen_labels.add(label)

        print(f"Total labels: {len(labels)}, Unique labels: {len(unique_labels)}")

        label_to_idx = {label: i for i, label in enumerate(unique_labels)}

        n = len(unique_labels)
        z_matrix = np.zeros((n - 1, 4))

        # Get the root of the tree
        if not clustering_builder.forest:
            print("ERROR: No tree found in the forest!")
            return z_matrix, unique_labels

        root = clustering_builder.forest[0]

        processed = {}  
        next_z_idx = n  
        row_idx = 0  

        def process_node(node):
            nonlocal row_idx, next_z_idx

            if node.node_id in processed:
                return processed[node.node_id]

            if len(node) == 0:  # No children
                if hasattr(node, "node_name") and node.node_name in label_to_idx:
                    idx = label_to_idx[node.node_name]
                    processed[node.node_id] = idx
                    return idx
                else:
                    return None

            left_idx = None
            right_idx = None

            if len(node) >= 1:
                left_idx = process_node(node[0])
            if len(node) >= 2:
                right_idx = process_node(node[1])

            if left_idx is None or right_idx is None:
                return None

            if left_idx > right_idx:
                left_idx, right_idx = right_idx, left_idx

            left_count = 1 if left_idx < n else z_matrix[left_idx - n][3]
            right_count = 1 if right_idx < n else z_matrix[right_idx - n][3]

            if row_idx < n - 1:  
                z_matrix[row_idx] = [
                    left_idx,
                    right_idx,
                    node.weight,
                    left_count + right_count,
                ]
                this_idx = next_z_idx
                processed[node.node_id] = this_idx
                next_z_idx += 1
                row_idx += 1
                return this_idx

            return None

        process_node(root)

        if row_idx < n - 1:
            print(f"WARNING: Only filled {row_idx} of {n-1} rows in Z matrix!")
            z_matrix = z_matrix[:row_idx]
            
        for i in range(z_matrix.shape[0]):
            left_idx = int(z_matrix[i, 0])
            right_idx = int(z_matrix[i, 1])

            left_size = 1 if left_idx < n else z_matrix[left_idx - n, 3]
            right_size = 1 if right_idx < n else z_matrix[right_idx - n, 3]
            expected_size = left_size + right_size
            
            if z_matrix[i, 3] != expected_size:
                print(
                    f"Fixing cluster size at row {i}: {z_matrix[i, 3]} → {expected_size}"
                )
                z_matrix[i, 3] = expected_size

            max_size = z_matrix.shape[0] + 1
            if z_matrix[i, 3] > max_size:
                print(
                    f"Capping excessive cluster size at row {i}: {z_matrix[i, 3]} → {max_size}"
                )
                z_matrix[i, 3] = max_size

        return z_matrix
