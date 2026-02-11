import numpy as np

class Node:
    def __init__(self, attr="", ans=""):
        self.attr = attr
        self.ans = ans
        self.children = []

def entropy(y):
    _, c = np.unique(y, return_counts=True)
    p = c / len(y)
    return -np.sum(p * np.log2(p)) if len(c) > 1 else 0

def gain_ratio(data, col):
    total_e = entropy(data[:, -1])
    vals, counts = np.unique(data[:, col], return_counts=True)
    split_info = -np.sum((counts/len(data)) * np.log2(counts/len(data)))
    
    weighted_e = 0
    for v in vals:
        sub = data[data[:, col] == v]
        weighted_e += (len(sub)/len(data)) * entropy(sub[:, -1])
    
    info_gain = total_e - weighted_e
    return info_gain / split_info if split_info else 0

def build(data, meta):
    y = data[:, -1]
    
    if len(set(y)) == 1:
        return Node(ans=y[0])
    if data.shape[1] == 1:
        vals, cnts = np.unique(y, return_counts=True)
        return Node(ans=vals[np.argmax(cnts)])

    gains = [gain_ratio(data, i) for i in range(data.shape[1]-1)]
    best = np.argmax(gains)
    node = Node(attr=meta[best])
    
    for v in np.unique(data[:, best]):
        sub = data[data[:, best] == v]
        sub = np.delete(sub, best, axis=1)
        node.children.append((v, build(sub, np.delete(meta, best))))
    
    return node

def print_tree(node, lvl=0):
    indent = "  " * lvl
    if node.ans:
        print(indent + "->", node.ans)
        return
    print(indent + f"[{node.attr}]")
    for v, child in node.children:
        print(indent + f" {v}:")
        print_tree(child, lvl+1)

# 🔥 READ FROM CSV FILE
# Make sure data.csv is in same folder

dataset = np.genfromtxt("data.csv", delimiter=",", dtype=str)

# First row = column names
meta = dataset[0][:-1]   # all columns except last
data = dataset[1:]       # remaining rows are data

# Build tree
root = build(data, meta)

print("\n--- DECISION TREE (C4.5 Gain Ratio) ---\n")
print_tree(root)
