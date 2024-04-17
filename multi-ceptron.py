import numpy as np
import matplotlib.pyplot as plt

LOSS_HISTORY_PATH = "visual_result/multi-ceptron-loss.png"
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 隨機初始化權重
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        
        # 初始化偏差
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        
        # 定義激活函數（sigmoid）
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.sigmoid_derivative = lambda x: x * (1 - x)
    
    def forward(self, inputs):
        self.hidden_sum = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_activation = self.sigmoid(self.hidden_sum)
        
        self.output_sum = np.dot(self.hidden_activation, self.weights_hidden_output) + self.bias_output
        self.output_activation = self.sigmoid(self.output_sum)
        
        return self.output_activation
    
    def backward(self, inputs, targets, learning_rate):
        output_error = targets - self.output_activation
        output_delta = output_error * self.sigmoid_derivative(self.output_activation)
        
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_activation)
        
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_activation.T, output_delta)
        self.weights_input_hidden += learning_rate * np.dot(inputs.T, hidden_delta)
        
        self.bias_output += learning_rate * np.sum(output_delta, axis=0)
        self.bias_hidden += learning_rate * np.sum(hidden_delta, axis=0)
    
    def train(self, inputs, targets, epochs, learning_rate):
        loss_history = []
        for epoch in range(epochs):
            self.forward(inputs)
            self.backward(inputs, targets, learning_rate)
            
            if epoch % 1000 == 0:
                loss = np.mean(np.square(targets - self.output_activation))
                loss_history.append(loss)
                print(f"Epoch {epoch}: Loss = {loss}")
        return loss_history

def test_nn(nn, inputs):
    print(f"Testing {nn.__class__.__name__}:")
    for i in range(len(inputs)):
        prediction = nn.forward(inputs[i])
        # set a threshold to classify the output
        # threshold = 0.5
        # classes = 0 if prediction < threshold else 1
        # print(f"Input: {inputs[i]}, Target: {targets[i]}, Prediction: {classes}")
        print(f"Input: {inputs[i]}, Target: {targets[i]}, Prediction: {prediction}")

if __name__ == "__main__":

    # ---------- test model (classification) ---------- #
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    # initial neural network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    # train the neural network
    loss_history = nn.train(inputs, targets, epochs=10000, learning_rate=0.1)

    # test the neural network
    test_nn(nn, inputs)

    # visual the loss history
    # plt.plot(loss_history)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.savefig(LOSS_HISTORY_PATH)