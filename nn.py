import numpy as np 

class NN:
    def __init__(self):
        # Initialise hash tables for parameters(weights and biases) and gradients(for backpropagation)
        self.grads = {} 
        self.params = {}
    
    def get_input(self, x):
        self.x = x
    
    def get_output(self, y):
        self.y = y

    # datastructure: [[input_size, output_size, activation], [output_size, output_size, activation], ...]
    def get_nn_architecture(self, architecture):
        self.architecture = architecture

    def init_params(self):
        pass