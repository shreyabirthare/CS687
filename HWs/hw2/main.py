########################################
# Python library requirements:
# - numpy
# - numba
########################################

from utils import *




if __name__ == "__main__":
    # Example of how to use the NeuralNetwork class

	
    # Create a neural network with 2 hidden layers, with 2 and 3 neurons respectively
    # Note: Whenever creating a new neural network, use the syntax below.
         # In particular, the number of layers has to be of type int64, and the number of neurons per layer has to be a numpy array of type int64
         # This will ensure that the optimizer (numba) will run as intended
    nn = NeuralNetwork(num_layers=np.int64(2), neurons_per_layer=np.array([2, 3], dtype=np.int64))

	# Example showing how to manually define a state of the MDP
    state = np.array([0, -0.5], dtype=np.float64)
    
    # Get the current weights of the network
    weights = nn.get_weights()

    # Create new random weights
    new_weights = np.random.randn(weights.shape[0])
	# Note: New weight vectors must be cast as a numpy array of type float64. Simply copy the line below after creating the weight vectors
    new_weights = new_weights.astype(np.float64)

    # Set the weights of the network to new values (i.e., upload the neural network's weights)
    nn.load_weights(new_weights)

    # Generate an action from the network given a state. This is the function that implements the policy, pi(state)
    print("Action: ", nn.get_action(state))