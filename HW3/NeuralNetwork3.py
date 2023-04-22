import sys, math
import numpy as np

class MyNeuralNetwork:
    def __init__(self, learning_rate, mini_batch_size, epochs, layers):
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.layers = layers
        self.parameters = {}

    # Generate mini-batches
    def mini_batches(self, X, y):
        m = X.shape[1]
        num_batches = math.floor(m / self.mini_batch_size)
        batches = []
        
        for i in range(num_batches):
            mini_X = X[:, i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            mini_y = y[:, i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            batches.append((mini_X, mini_y))
        
        if m % self.mini_batch_size != 0:
            mini_X = X[:, num_batches * self.mini_batch_size : m]
            mini_y = y[:, num_batches * self.mini_batch_size : m]
            batches.append((mini_X, mini_y))
        
        return batches
    
    # Initialize weights and biases for each layer
    def init_weights_biases(self):
        np.random.seed(42)
        self.parameters['W1'] = np.random.randn(self.layers[1], self.layers[0]) * np.sqrt(1 / self.layers[0])
        self.parameters['b1'] = np.random.randn(self.layers[1], 1)

        self.parameters['W2'] = np.random.randn(self.layers[2], self.layers[1]) * np.sqrt(1 / self.layers[1])
        self.parameters['b2'] = np.random.randn(self.layers[2], 1)

        self.parameters['W3'] = np.random.randn(self.layers[3], self.layers[2]) * np.sqrt(1 / self.layers[2])
        self.parameters['b3'] = np.random.randn(self.layers[3], 1)
    
    # Use the sigmoid function as the activation function
    def sigmoid(self, x):
        res = 1 / (1 + np.exp(-x))
        return res
    
    # Compute derivative of the sigmoid function for backpropagation
    def sigmoid_derivative(self, x):
        res = x * (1 - x)
        return res
    
    # Calculate the cross-entropy loss
    def cross_entropy_loss(self, y_true, y_pred):
        m = y_pred.shape[1]
        loss = - (np.dot(y_true, np.log(y_pred).T) + np.dot(1 - y_true, np.log(1 - y_pred).T)) / m
        return loss

    # Forward propagation with 2 hidden layers
    def forward_propagation(self, X):
        Z1 = np.dot(self.parameters['W1'], X) + self.parameters['b1']
        A1 = self.sigmoid(Z1)

        Z2 = np.dot(self.parameters['W2'], A1) + self.parameters['b2']
        A2 = self.sigmoid(Z2)

        Z3 = np.dot(self.parameters['W3'], A2) + self.parameters['b3']
        A3 = self.sigmoid(Z3)
                
        cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}
        return cache, A3
    
    # Backpropagation to update weights and biases
    def backpropagation(self, X, y, cache):
        m = X.shape[1]
        
        dA3 = cache['A3'] - y
        dZ3 = dA3 * self.sigmoid_derivative(cache['A3'])
        dW3 = np.dot(dZ3, cache['A2'].T) / m
        db3 = np.sum(dZ3, axis = 1, keepdims = True) / m
        
        dA2 = np.dot(self.parameters['W3'].T, dZ3)
        dZ2 = dA2 * self.sigmoid_derivative(cache['A2'])
        dW2 = np.dot(dZ2, cache['A1'].T) / m
        db2 = np.sum(dZ2, axis = 1, keepdims = True) / m
        
        dA1 = np.dot(self.parameters['W2'].T, dZ2)
        dZ1 = dA1 * self.sigmoid_derivative(cache['A1'])
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis = 1, keepdims = True) / m
        
        self.parameters['W3'] = self.parameters['W3'] - self.learning_rate * dW3
        self.parameters['b3'] = self.parameters['b3'] - self.learning_rate * db3
        
        self.parameters['W2'] = self.parameters['W2'] - self.learning_rate * dW2
        self.parameters['b2'] = self.parameters['b2'] - self.learning_rate * db2
        
        self.parameters['W1'] = self.parameters['W1'] - self.learning_rate * dW1
        self.parameters['b1'] = self.parameters['b1'] - self.learning_rate * db1
    
    # Fit the training data
    def fit(self, X_train, y_train):
        self.init_weights_biases()
        
        for i in range(self.epochs):        
            batches = self.mini_batches(X_train, y_train)
            for batch in batches:
                mini_x, mini_y = batch[0], batch[1]
                cache, A3 = self.forward_propagation(mini_x)
                # loss = self.cross_entropy_loss(mini_y, A3)
                # print('loss:', loss)
                self.backpropagation(mini_x, mini_y, cache)

    # Predict the output
    def predict(self, X):
        Z1 = np.dot(self.parameters['W1'], X) + self.parameters['b1']
        A1 = self.sigmoid(Z1)

        Z2 = np.dot(self.parameters['W2'], A1) + self.parameters['b2']
        A2 = self.sigmoid(Z2)

        Z3 = np.dot(self.parameters['W3'], A2) + self.parameters['b3']
        A3 = self.sigmoid(Z3)
        
        y_pred = (A3 > 0.5)
        return y_pred

    # Calculate the accuracy
    def accuracy(self, y_true, y_pred):
        acc = sum(y_true.squeeze() == y_pred.squeeze()) / len(y_true.squeeze())
        return acc

train_data_filename = sys.argv[1]
train_data = np.loadtxt(train_data_filename, delimiter = ',', ndmin = 2)

train_label_filename = sys.argv[2]
train_label = np.loadtxt(train_label_filename, delimiter = ',', ndmin = 2)

test_data_filename = sys.argv[3]
test_data = np.loadtxt(test_data_filename, delimiter = ',', ndmin = 2)

# test_label_filename = 'xxx_test_label.csv'
# test_label = np.loadtxt(test_data_filename, delimiter = ',', ndmin = 2)

X_train = train_data.T
y_train = train_label.T
X_test = test_data.T
# y_test = test_label.T

learning_rate = 0.3
mini_batch_size = 32
epochs = 2000
layers = [2, 100, 50, 1]
clf = MyNeuralNetwork(learning_rate, mini_batch_size, epochs, layers)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# acc = clf.accuracy(y_test, y_pred)
# print('accuracy:' acc)

np.savetxt('test_predictions.csv', y_pred.T, newline = '\n', fmt = '%i')