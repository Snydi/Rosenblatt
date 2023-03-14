import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron as sklearn_Perceptron
from sklearn.preprocessing import StandardScaler

class Perceptron:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def fit(self, X, y):
        # Initialize weights to zeros
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                # Calculate the predicted value
                y_pred = self.predict(X[i])
                
                # Update the weights and bias if prediction is incorrect
                if y[i]*y_pred <= 0:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]
                    
    def predict(self, x):
        # Calculate the dot product of weights and input
        z = np.dot(x, self.weights) + self.bias
        
        # Return the sign of the dot product
        return np.sign(z)
    
def main():
    # Load the data from CSV file
    data = pd.read_csv("data.csv")
    
    # Split the data into features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Convert labels to -1 or 1
    y = np.where(y == 0, -1, 1)
    
    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Initialize the Perceptron object
    perceptron = Perceptron(learning_rate=0.1, epochs=100)
    
    # Train the Perceptron model
    perceptron.fit(X, y)
    
    # Predict the labels for the input data
    y_pred = np.array([perceptron.predict(x) for x in X])
    
    # Print the learned weights, bias, and predicted labels
    print("Learned Weights (From Scratch):", perceptron.weights)
    print("Learned Bias (From Scratch):", perceptron.bias)
    print("Predicted Labels (From Scratch):", y_pred)
    
    # Compare with the Perceptron model implemented
    # Compare with the Perceptron model implemented in scikit-learn
    sklearn_perceptron = sklearn_Perceptron(alpha=0, max_iter=100, tol=None)
    sklearn_perceptron.fit(X, y)
    
    # Predict the labels using the scikit-learn Perceptron model
    sklearn_y_pred = sklearn_perceptron.predict(X)
    
    # Print the learned weights, bias, and predicted labels using scikit-learn
    print("Learned Weights (Using Scikit-Learn):", sklearn_perceptron.coef_[0])
    print("Learned Bias (Using Scikit-Learn):", sklearn_perceptron.intercept_[0])
    print("Predicted Labels (Using Scikit-Learn):", sklearn_y_pred)
    
    # Calculate the accuracy of the models
    acc_from_scratch = sum(y == y_pred) / len(y)
    acc_scikit_learn = sum(y == sklearn_y_pred) / len(y)
    
    # Print the accuracies of the models
    print("Accuracy (From Scratch):", acc_from_scratch)
    print("Accuracy (Using Scikit-Learn):", acc_scikit_learn)
    
if __name__ == "__main__":
    main()
