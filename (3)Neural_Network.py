from math import exp
from random import seed, random

def initialize_network(n_inputs, n_hidden, n_outputs):
    return [
        [{'weights': [random() for _ in range(n_inputs + 1)]} for _ in range(n_hidden)],
        [{'weights': [random() for _ in range(n_hidden + 1)]} for _ in range(n_outputs)]
    ]

def activate(weights, inputs):
    return sum(w*i for w, i in zip(weights[:-1], inputs)) + weights[-1]

def transfer(x):
    return 1.0 / (1.0 + exp(-x))

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        inputs = [transfer(activate(n['weights'], inputs)) for n in layer]
        for n, out in zip(layer, inputs):
            n['output'] = out
    return inputs

def transfer_derivative(output):
    return output * (1 - output)

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        if i == len(network) - 1:
            errors = [n['output'] - expected[j] for j, n in enumerate(layer)]
        else:
            errors = [
                sum(n['weights'][j] * n['delta'] for n in network[i+1])
                for j in range(len(layer))
            ]
        for j, n in enumerate(layer):
            n['delta'] = errors[j] * transfer_derivative(n['output'])

def update_weights(network, row, l_rate):
    for i, layer in enumerate(network):
        inputs = row[:-1] if i == 0 else [n['output'] for n in network[i-1]]
        for n in layer:
            for j in range(len(inputs)):
                n['weights'][j] -= l_rate * n['delta'] * inputs[j]
            n['weights'][-1] -= l_rate * n['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row[:-1])
            expected = [0]*n_outputs
            expected[row[-1]] = 1
            sum_error += sum((e-o)**2 for e, o in zip(expected, outputs))
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print(f'>epoch={epoch}, lrate={l_rate:.3f}, error={sum_error:.3f}')

# Test
seed(1)
dataset = [
    [2.7810836,2.550537003,0],[1.465489372,2.362125076,0],
    [3.396561688,4.400293529,0],[1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],[7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],[6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],[7.673756466,3.508563011,1]
]

n_inputs = len(dataset[0]) - 1
n_outputs = len(set(row[-1] for row in dataset))

network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)

for layer in network:
    print(layer)
