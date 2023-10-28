import numpy as np
import os 
import joblib


class perceptron:
    def __init__(self, eta: float=None, epochs: int=None):
        # Initialize Perceptron with small random weights
        self.weights = np.random.randn(3) * 1e-4 
        # Check if training parameters are provided and print initial weights
        training = (eta is not None) and (epochs is not None)
        if training:
            print(f"initial weights before trining: \n {self.weights}")
        # Set learning rate (eta) and number of epochs
        self.eta = eta
        self.epochs = epochs

    def _z_outcome(self, inputs, weights):
        # Calculate the weighted sum of inputs
        return np.dot(inputs, weights)
        
    def activation_function(self, z):
        # Apply activation function (binary step function)
        return np.where(z > 0, 1, 0) 

    def fit(self, X, y):
        # Store input data and target labels
        self.X = X
        self.y = y

        # Add a bias column to input data
        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
        print(f"X with bias: \n{X_with_bias}")

        for epoch in range(self.epochs):
            print("--"*10)
            print(f"for epoch >> {epoch}")
            print("--"*10)

            # Calculate the weighted sum and apply the activation function
            z = self._z_outcome(X_with_bias, self.weights)
            y_hat = self.activation_function(z)
            print(f"predicted value after forward pass: \n{y_hat}")

            # Calculate the prediction error
            self.error = self.y - y_hat
            print(f"error: \n{self.error}")

            # Update the weights using the error and learning rate
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error)
            print(f"updated weights after epoch: {epoch + 1}/{self.epochs}: \n{self.weights}")
            print("##"*10)
            

    def predict(self, X):
        # Add a bias column to input data and make predictions
        X_with_bias = np.c_[X, -np.ones((len(X), 1))]
        z = self._z_outcome(X_with_bias, self.weights)
        return self.activation_function(z)

    def total_loss(self):
        # Calculate and print the total loss (sum of errors)
        total_loss = np.sum(self.error)
        print(f"\ntotal loss: {total_loss}\n")
        return total_loss

    def _create_dir_return_path(self, model_dir, filename):
        # Create a directory and return the path
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, filename)

    def save(self, filename, model_dir= None):
        if model_dir is not None:
            # If a model directory is specified, save the model there
            model_file_path = self._create_dir_return_path(model_dir, filename)
            joblib.dump(self, model_file_path)
        else:
            # Otherwise, save the model in the "model" directory
            model_file_path = self._create_dir_return_path("model", filename)
            joblib.dump(self, model_file_path)

    def load(self, filepath):
        # Load a saved model from the specified file path (not defined in the code)
        return joblib.load(filepath)