import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(0)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def spiral_data(samples, classes):
    return [np.random.randn(samples, 2),
            np.random.randint(0,classes,(samples))]

def gen_data(m):
    x = np.random.randn(m,2)
    y_true = np.zeros((m,2))
    for i in range(0,m):
        y_true[i][int((x[i][0]<x[i][1]))] = 1 
    return x, y_true

def gen_mnist(i, batch):
    x = np.array(x_train[batch*i:batch*(i+1)])
    y = np.array(y_train[batch*i:batch*(i+1)])
    
    x=x.astype(float)
    
    x = np.reshape(x, (batch,784))
    y = np.reshape(y, batch)
    return x, y

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    def update(self, dW, dB):
        self.weights = self.weights - dW
        self.biases = self.biases - dB
    
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    def backward(self, inputs):
        return inputs>0
        
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, L):
        return -1/np.exp(-L)

def backprop(X, y_true, layer1, layer2, activation1, m):
    d=0.1
    
    if len(y_true.shape) > 1:
        y = np.zeros((m, len(layer2.output[0])))
        y_true = y[range(m), y_true]
    
    softmax = Activation_Softmax()
    relu = Activation_ReLU()
    
    softmax.forward(layer2.output)
    dZ2 = (softmax.output-y_true)
    dB2 = 1/m*np.sum(dZ2, axis=0, keepdims=True)
    dW2 = 1/m*activation1.output.T.dot(dZ2)
    
    reluback = relu.backward(layer1.output)
    dZ1 = dZ2.dot(layer2.weights.T)
    dZ1 = np.multiply(dZ1, reluback)
    
    dB1 = 1/m*np.sum(dZ1, axis=0, keepdims=True)
    dW1 = 1/m*X.T.dot(dZ1)

    layer1.update(d*dW1, d*dB1)
    layer2.update(d*dW2, d*dB2)

losses = []

dense1 = Layer_Dense(784, 30)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(30, 10)
activation2 = Activation_Softmax()

for i in range(0,1000):
    m = 30
    X, y = gen_mnist(i,m)

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss_function = Loss_CategoricalCrossentropy()
    loss = loss_function.calculate(activation2.output, y)
    losses.append(loss)
    
    backprop(X, y, dense1, dense2, activation1, m)

     
plt.plot(losses)

#plt.plot(dense2.weights[0][1])
    
