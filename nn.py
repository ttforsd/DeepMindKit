import numpy as np 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
class NN:
    def __init__(self):
        # Initialise hash tables for parameters(weights and biases) and gradients(for backpropagation)
        self.grads = {"w": [], "b": [], "z": [], "a": []}
        self.params = {"w": [], "b": []}
        self.cache = {"z": [], "a": []}

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    
    def d_sigmoid_dx(self, x):
        return x * (1 - x)
    
    def d_relu_dx(self, x):
        return (x > 0).astype(int)
    
    def d_tanh_dx(self, x):
        return 1 - x**2
    
    
    def get_input(self, x):
        self.x = x
    
    def get_output(self, y):
        self.y = y

    # datastructure: [[input_size, output_size, activation], [input_size, output_size, activation], ...]
    def get_nn_architecture(self, architecture):
        self.architecture = architecture
        self.init_params()

    def init_params(self):
        for layer in self.architecture:
            # w : row vectors, avoid transpoing in forw_prop
            self.params["w"].append(np.random.randn(layer[1], layer[0]) * 0.01)
            self.params["b"].append(np.zeros((layer[1], 1)))
            # cache for backpropagation 
            # gradients of loss with respect to parameters
            self.grads["w"].append(np.zeros((layer[1], layer[0])))
            self.grads["b"].append(np.zeros((layer[1], 1)))
            # gradients of loss with respect to Z (linear) and A (activation) for each layer
            self.grads["z"].append(np.zeros((layer[1], 1)))
            self.grads["a"].append(np.zeros((layer[1], 1)))
            # add dim to cache 
            self.cache["z"].append(None)
            self.cache["a"].append(None)
        # print(self.params)
        # print(self.grads)

    def loss_func(self, y_hat, y, func="cross_entropy"):
        m = y.shape[1]
        if func == "cross_entropy":
            res = np.dot(y, np.log(y_hat + 1e-8).T) + np.dot(1 - y, np.log(1 - y_hat + 1e-8).T)
            res /= -m 
            return float(res)
        elif func == "mse":
            return np.sum((y - y_hat)**2) / m 
        else: 
            print("Cost function not implemented")

    def d_cost_d_y_hat(self, y_hat, y, func="cross_entropy"):
        if func == "cross_entropy":
            res = (y_hat - y) / (y_hat * (1 - y_hat) + 1e-8)
            return res
        elif func == "mse":
            return -2 * (y - y_hat)
        else: 
            print("Cost function not implemented")

    def forw_prop(self, x): 
        res = x
        for i in range(len(self.params["w"])):
            res = np.dot(self.params["w"][i], res) + self.params["b"][i]
            self.cache["z"][i] = res[:]
            # multiplication of w row vectors and previous layer give shape (n, m) where n is the number of neurons in the current layer and m is the number of samples
            if self.architecture[i][2] == "relu":
                res = self.relu(res)
            elif self.architecture[i][2] == "sigmoid":
                res = self.sigmoid(res)
            elif self.architecture[i][2] == "tanh":
                res = self.tanh(res)
            self.cache["a"][i] = res[:]
        return res 
    
    def back_prop(self, y_hat, y, x): 
        # iterate in reverse order
        for i in range(1, len(self.architecture) + 1):
            i = -i # reverse index
            if i == -1: # last layer 
                # update dL/dy_hat 
                self.grads["a"][i] = self.d_cost_d_y_hat(y_hat, y, func="cross_entropy")
            else: 
                # update dL/da
                self.grads["a"][i] = np.dot(self.params["w"][i + 1].T, self.grads["z"][i + 1])
            # update dL/dz
            if self.architecture[i][2] == "relu":
                self.grads["z"][i] = self.grads["a"][i] * self.d_relu_dx(self.cache["a"][i])
            elif self.architecture[i][2] == "sigmoid":
                self.grads["z"][i] = self.grads["a"][i] * self.d_sigmoid_dx(self.cache["a"][i])
            elif self.architecture[i][2] == "tanh":
                self.grads["z"][i] = self.grads["a"][i] * self.d_tanh_dx(self.cache["a"][i])
            
            # update dL/dw
            if -i == len(self.architecture):
                self.grads["w"][i] = np.dot(self.grads["z"][i], x.T)

            else: 
                self.grads["w"][i] = np.dot(self.grads["z"][i], self.cache["a"][i - 1].T) / x.shape[1]
            self.grads["b"][i] = self.grads["z"][i]
            self.grads["b"][i] = np.sum(self.grads["b"][i], axis=1, keepdims=True) / x.shape[1]
    
    def train(self, batch_size=32, epochs=50, lr=0.01): 
        for epoch in range(epochs):
            for i in range(0, self.x.shape[1], batch_size):
                end = min(i + batch_size, self.x.shape[1])
                batch_m = end - i
                x_batch = self.x[:, i:end]
                y_batch = self.y[:, i:end]
                y_hat = self.forw_prop(x_batch)
                self.back_prop(y_hat, y_batch, x_batch)
                for j in range(len(self.params["w"])):
                    self.params["w"][j] -= lr * self.grads["w"][j] 
                    self.params["b"][j] -= lr * self.grads["b"][j] 

            print(f"Epoch {epoch} loss: ", self.loss_func(self.forw_prop(self.x), self.y))

    def predict(self):
        return self.forw_prop(self.x)
    def save_params(self, path):
        with open(path, "wb") as f:
            f.write(self.params)


def tests(): 
    data_path = "./data/breast_cancer.csv"
    data = pd.read_csv(data_path)
    data.info()
    Y = data['diagnosis']
    X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
    Y = Y.map({'M': True, 'B': False})
# split data into train, test, validation (70, 20, 10) randomly 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.33, random_state=42)

    # normalize data
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    X_val = (X_val - mean) / std

    X_train = X_train.values
    Y_train = Y_train.values
    X_test = X_test.values
    Y_test = Y_test.values
    X_val = X_val.values
    Y_val = Y_val.values
    print("Train data shape: ", X_train.shape)
    print("Test data shape: ", X_test.shape)
    print("Validation data shape: ", X_val.shape)
# neurons: 10 relu -> 5 relu -> 1 sigmoid
    input_dim = X_train.shape[1]
    print("Input dimension: ", input_dim)
    output_dim = 1
    hidden_dims = [10, 5, 1]
    arch = [[input_dim, hidden_dims[0], 'relu'], [hidden_dims[0], hidden_dims[1], 'relu'], [hidden_dims[1], output_dim, 'sigmoid']]
    my_model = NN()
    my_model.get_input(X_train.T) 
    # make sure Y_train is 2D
    Y_train_2d = Y_train.reshape(-1, 1)
    my_model.get_output(Y_train_2d.T)

    my_model.get_nn_architecture(arch)

    my_model.train(epochs=180, lr=0.01, batch_size=32)

    # test on validation set
    y_pred = my_model.predict()
    y_pred = y_pred > 0.5
    print("Accuracy: ", np.mean(y_pred == Y_train_2d.T))
    # test on test set
    my_model.get_input(X_test.T)
    my_model.get_output(Y_test.reshape(-1, 1).T)
    y_pred = my_model.predict()
    y_pred = y_pred > 0.5
    print("Accuracy (test set): ", np.mean(y_pred == Y_test.reshape(-1, 1).T))

    # test on validation set
    my_model.get_input(X_val.T)
    my_model.get_output(Y_val.reshape(-1, 1).T)
    y_pred = my_model.predict()
    y_pred = y_pred > 0.5
    print("Accuracy (validation set): ", np.mean(y_pred == Y_val.reshape(-1, 1).T))




if __name__ == "__main__":
    model = NN() 
    y = np.array([[1, 0, 1, 0, 1]])
    y_hat = np.array([[1, 0, 1, 0, 1]])
    print(model.loss_func(y_hat, y))
    tests()
    # test = NN()
    # # input dim: 2, m = 10
    # x = np.random.randn(2, 10)
    # print(x)
    # test.get_input(x)
    # # test forw_prop
    # test.get_output(np.random.randn(1, 10))
    # test.get_nn_architecture([[2, 3, "relu"], [3, 1, "sigmoid"]])
    # print(test.architecture)


                         
    