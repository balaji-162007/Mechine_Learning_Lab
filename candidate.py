import numpy as np
import pandas as pd

# Read data from CSV into a DataFrame
data = pd.read_csv("trainingdata.csv")
print(data)

# Extract features (concepts) and labels (target)
concepts = np.array(data.iloc[:, :-1])
print(concepts)

target = np.array(data.iloc[:, -1])
print(target)

# Candidate Elimination Algorithm
def learn(concepts, target):
    # Initialize specific hypothesis
    specific_h = concepts[0].copy()
    print("\nInitialization of specific_h and general_h")
    print("Specific_h:", specific_h)

    # Initialize general hypothesis
    general_h = [["?" for _ in range(len(specific_h))]
                 for _ in range(len(specific_h))]
    print("General_h:", general_h)

    # Iterate through training instances
    for i, h in enumerate(concepts):
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = "?"
                    general_h[x][x] = "?"

        elif target[i] == "No":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = "?"

        print(f"\nStep {i + 1}")
        print("Specific_h:", specific_h)
        print("General_h:", general_h)

    # Remove overly general hypotheses
    general_h = [h for h in general_h if h != ["?"] * len(specific_h)]

    return specific_h, general_h


# Apply algorithm
s_final, g_final = learn(concepts, target)

print("\nFinal Specific_h:")
print(s_final)

print("\nFinal General_h:")
print(g_final)
