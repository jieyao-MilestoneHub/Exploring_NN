import numpy as np

class Perceptron:
    def __init__(self, input_size, activation_function, learning_rate=0.01):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)
        self.activation_function = activation_function
        self.learning_rate = learning_rate
    
    def activate(self, x):
        return self.activation_function(x)
    
    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activate(weighted_sum)
    
    def train(self, training_inputs, training_outputs, epochs):
        for _ in range(epochs):
            for inputs, output in zip(training_inputs, training_outputs):
                prediction = self.forward(inputs)
                error = output - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

# Activation functions for AND and OR gates for binary classification
def and_gate_activation(x):
    return 1 if x >= 0 else 0

def or_gate_activation(x):
    return 0 if x < 0.5 else 1

# Standard activation function
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

# Function to test gates
def test_gate(gate, inputs):
    print(f"Testing {gate.__class__.__name__}:")
    for input_data in inputs:
        print(f"{input_data} -> {gate.forward(input_data)}")


if __name__ == "__main__":
    # Initialize and train Perceptrons
    and_gate = Perceptron(2, and_gate_activation, 0.1)
    and_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_outputs = np.array([0, 0, 0, 1])
    and_gate.train(and_inputs, and_outputs, 1000)

    or_gate = Perceptron(2, or_gate_activation, 0.1)
    or_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    or_outputs = np.array([0, 1, 1, 1])
    or_gate.train(or_inputs, or_outputs, 1000)

    # Initialize and train a Perceptron with Sigmoid activation
    perceptron = Perceptron(2, sigmoid_function, 0.1)
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_outputs = np.array([0, 1, 1, 1])
    perceptron.train(training_inputs, training_outputs, 1000)

    # Test all gates and perceptrons
    test_gate(and_gate, and_inputs)
    test_gate(or_gate, or_inputs)
    test_gate(perceptron, training_inputs)
