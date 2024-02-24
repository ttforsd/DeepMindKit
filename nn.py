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
    
    
    def d_sigmoid(self, x):
        return x * (1 - x)
    
    def d_relu(self, x):
        return 1 if x > 0 else 0
    
    def d_tanh(self, x):
        return 1 - x**2
    
    
    def get_input(self, x):
        self.x = x
    
    def get_output(self, y):
        self.y = y

    # datastructure: [[input_size, output_size, activation], [output_size, output_size, activation], ...]
    def get_nn_architecture(self, architecture):
        self.architecture = architecture
        self.init_params()

    def init_params(self):
        for layer in self.architecture:
            self.params["w"].append(np.random.randn(layer[0], layer[1])/ 100)
            self.params["b"].append(np.zeros((layer[1], 1)))
            self.grads["w"].append(np.zeros((layer[0], layer[1])))
            self.grads["b"].append(np.zeros(layer[1]))
        print(self.params)
        print(self.grads)


    def forw_prop(self): 
        res = self.x 
        for i in range(len(self.params["w"])):
            res = np.dot(self.params["w"][i].T, res) + self.params["b"][i]
            if self.architecture[i][2] == "relu":
                res = self.relu(res)
            elif self.architecture[i][2] == "sigmoid":
                res = self.sigmoid(res)
            elif self.architecture[i][2] == "tanh":
                res = self.tanh(res)
        return res 





test = NN()
# input dim: 2, m = 10
x = np.random.randn(2, 10)
print(x)
test.get_input(x)
# test forw_prop
test.get_nn_architecture([[2, 3, "relu"], [3, 1, "sigmoid"]])
print(test.forw_prop())


                         
    