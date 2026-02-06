import numpy as np

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""

def subtables(data, col, delete=False):
    items = np.unique(data[:, col])
    res_dict = {}
    for item in items:
        # Filter rows where the column matches the current item
        sub_data = data[data[:, col] == item]
        if delete:
            sub_data = np.delete(sub_data, col, 1)
        res_dict[item] = sub_data
    return items, res_dict

def entropy(S):
    items, counts = np.unique(S, return_counts=True)
    if items.size <= 1:
        return 0
    
    probs = counts / S.size
    return -np.sum(probs * np.log2(probs))

def gain_ratio(data, col):
    items, dict_sub = subtables(data, col, delete=False)
    total_size = data.shape[0]
    total_entropy = entropy(data[:, -1])
    
    weighted_entropy = 0
    intrinsic_val = 0
    
    for item in items:
        sub_group = dict_sub[item]
        ratio = sub_group.shape[0] / total_size
        weighted_entropy += ratio * entropy(sub_group[:, -1])
        intrinsic_val -= ratio * np.log2(ratio)
        
    info_gain = total_entropy - weighted_entropy
    # Gain Ratio = Info Gain / Split Info (intrinsic value)
    return info_gain / intrinsic_val if intrinsic_val != 0 else 0

def create_node(data, metadata):
    # Case 1: All targets are the same
    if len(np.unique(data[:, -1])) == 1:
        node = Node("")
        node.answer = np.unique(data[:, -1])[0]
        return node
    
    # Case 2: No more features left to split
    if data.shape[1] == 1:
        node = Node("")
        vals, counts = np.unique(data[:, -1], return_counts=True)
        node.answer = vals[np.argmax(counts)]
        return node

    # Calculate Gain Ratio for each attribute
    gains = [gain_ratio(data, col) for col in range(data.shape[1] - 1)]
    split = np.argmax(gains)
    
    node = Node(metadata[split])
    items, dict_sub = subtables(data, split, delete=True)
    
    # Update metadata for the child nodes
    new_metadata = np.delete(metadata, split, 0)
    
    for item in items:
        child = create_node(dict_sub[item], new_metadata)
        node.children.append((item, child))
    return node

def print_tree(node, level=0):
    indent = "    " * level
    if node.answer != "":
        print(f"{indent}--> {node.answer}")
        return
    print(f"{indent}[{node.attribute}]")
    for value, n in node.children:
        print(f"{indent}  {value}:")
        print_tree(n, level + 1)

# --- Integrated Dataset ---
metadata = np.array(["Outlook", "Temperature", "Humidity", "Wind"])

# Format: [Outlook, Temp, Humidity, Wind, PlayTennis (Target)]
traindata = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

data = np.array(traindata)

# Build and Print
root = create_node(data, metadata)
print("\n--- DECISION TREE (C4.5 Gain Ratio) ---\n")
print_tree(root)
