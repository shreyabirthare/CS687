from numba import float64, int64
from numba.experimental import jitclass
import numpy as np

spec = [
    ('num_layers', int64), 
    ('neurons_per_layer', int64[:]),
    ('flat_weights', float64[:]),
    ('flat_biases', float64[:]),
]

@jitclass(spec)
class NeuralNetwork:
    def __init__(self, num_layers, neurons_per_layer):
        self.num_layers = num_layers + 1  # Number of hidden layers + output layer
        self.neurons_per_layer = np.concatenate((np.array([3]), neurons_per_layer, np.array([1])))  # input layer + hidden layers + output layer
        self.flat_weights, self.flat_biases = self._initialize_weights_and_biases()
    
    # Initialize weights and biases with random values
    def _initialize_weights_and_biases(self):
        # Calculate the total number of weights and biases, based on the number of neurons in each layer
        num_weights = 0
        num_biases = 0
        for i in range(1, len(self.neurons_per_layer)):
            num_weights += self.neurons_per_layer[i-1] * self.neurons_per_layer[i]
            num_biases += self.neurons_per_layer[i]

        # Return a tuple of two arrays: one for weights and one for biases, both containing random values
        return np.random.randn(num_weights), np.random.randn(num_biases) * 0.01
    
    # Activation function (tanh)
    def _activation(self, z):
        return np.tanh(z)
        
    # Unflatten the weights and biases from a flat array to a list of arrays (one for each layer) with the correct shape
    def _unflatten_weights_and_biases(self):
        weights = []
        biases = []

        start_w = 0
        start_b = 0
        for i in range(1, len(self.neurons_per_layer)):
            w_size = self.neurons_per_layer[i-1] * self.neurons_per_layer[i]
            layer_weights = np.copy(self.flat_weights[start_w:start_w+w_size])
            layer_weights = layer_weights.reshape((self.neurons_per_layer[i-1], self.neurons_per_layer[i]))
            weights.append(layer_weights)

            start_w += w_size
            b_size = self.neurons_per_layer[i]
            layer_biases = np.copy(self.flat_biases[start_b:start_b+b_size])
            layer_biases = layer_biases.reshape((self.neurons_per_layer[i]))
            biases.append(layer_biases)
            start_b += b_size

        return weights, biases

    # Forward pass through the network: takes a state as input and returns an action (the output of the network)
    def forward(self, state):
        features = np.array([np.cos(state[0]), np.sin(state[0]), state[1]])

        weights, biases = self._unflatten_weights_and_biases()

        a = features
        for i in range(self.num_layers):
            z = np.dot(a, weights[i]) + biases[i]
            if i < self.num_layers - 1:
                a = self._activation(z)
        return z
    
    # Returns the current weights and biases of the network as a single array
    def get_weights(self):
        return np.concatenate((self.flat_weights, self.flat_biases))
    
    # Load new weights and biases into the network from a single array
    def load_weights(self, all_weights):
        num_weights = self.flat_weights.size
        self.flat_weights = all_weights[:num_weights]
        self.flat_biases = all_weights[num_weights:]
        
    # Get the action produced by the network for a given state
    def get_action(self, state):
        return np.tanh(self.forward(state)[0]) * 2.0