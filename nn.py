import numpy as np 

class NN:
    def __init__(self):
        # Initialise hash tables for parameters(weights and biases) and gradients(for backpropagation)
        self.grads = {"w": [], "b": []}
        self.params = {"w": [], "b": []}

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    
    def d_sigmoid_dx(self, x):
        return x * (1 - x)
    
    def d_relu_dx(self, x):
        return 1 if x > 0 else 0
    
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
        print(self.params)
        print(self.grads)

    def loss_func(self, y_hat, y, func="cross_entropy"):
        if func == "cross_entropy":
            return -np.sum(self.y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) 
        elif func == "mse":
            return np.sum((y - y_hat)**2)
        else: 
            print("Cost function not implemented")

    def d_cost_d_y_hat(self, y_hat, func="cross_entropy"):
        if func == "cross_entropy":
            return - (self.y / y_hat - (1 - self.y) / (1 - y_hat))
        elif func == "mse":
            return -2 * (self.y - y_hat)
        else: 
            print("Cost function not implemented")

    def forw_prop(self, x): 
        res = x
        self.cache = {"z": [], "a": []}
        for i in range(len(self.params["w"])):
            res = np.dot(self.params["w"][i], res) + self.params["b"][i]
            self.cache["z"].append(res[:])
            # multiplication of w row vectors and previous layer give shape (n, m) where n is the number of neurons in the current layer and m is the number of samples
            if self.architecture[i][2] == "relu":
                res = self.relu(res)
            elif self.architecture[i][2] == "sigmoid":
                res = self.sigmoid(res)
            elif self.architecture[i][2] == "tanh":
                res = self.tanh(res)
            self.cache["a"].append(res[:])
        return res 
    
    def back_prop(self, y_hat, y): 
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
            self.grads["w"][i] = np.dot(self.grads["z"][i], self.cache["a"][i - 1].T)
            # update dL/db
            self.grads["b"][i] = self.grads["z"][i]
    
    def train(self, batch_size=32, epochs=50, lr=0.01): 
        for epoch in range(epochs):
            for i in range(0, self.x.shape[1], batch_size):
                end = max(i + batch_size, self.x.shape[1])
                batch_m = end - i
                x_batch = self.x[:, i:end]
                y_batch = self.y[:, i:end]
                y_hat = self.forw_prop(x_batch)
                self.back_prop(y_hat, y_batch)
                for i in range(len(self.params["w"])):
                    self.params["w"][i] -= lr * self.grads["w"][i] / batch_m
                    self.params["b"][i] -= lr * self.grads["b"][i] / batch_m

    def predict(self):
        return self.forw_prop(self.x)
    def save_params(self, path):
        with open(path, "wb") as f:
            f.write(self.params)

test = NN()
# input dim: 2, m = 10
x = np.random.randn(2, 10)
print(x)
test.get_input(x)
# test forw_prop
test.get_output(np.random.randn(1, 10))
test.get_nn_architecture([[2, 3, "relu"], [3, 1, "sigmoid"]])
print(test.forw_prop())

print(test.cost(test.forw_prop(), func="cross_entropy"))


                         
    