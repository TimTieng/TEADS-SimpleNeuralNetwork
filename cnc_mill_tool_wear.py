# Tech Excellence Advance Data Science
# In-Class Group Work - Build a perceptron neural network
# Group Members - Tim Tieng, Kara Probosco,Nicholas Royal, Arvind Krishnan

# Import Required Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn
import os

# Check working directory, change if in different dir where data is
print(f"Current Dir: {os.getcwd()}")

data_dir = os.getcwd() + "/CNC Dataset"
os.chdir(data_dir)
# Check
print(f"Current Dir: {os.getcwd()}")

# Join experiments in a single dataframe and add tool condition column
frames = list()
results = pd.read_csv("train.csv")
for i in range(1,19):
    exp = '0' + str(i) if i < 10 else str(i)# concatenate i up to 09, then just concat value
    frame = pd.read_csv("experiment_{}.csv".format(exp))
    row = results[results['No'] == i]
    frame['target'] = 1 if row.iloc[0]['tool_condition'] == 'worn' else 0
    frames.append(frame)
df = pd.concat(frames, ignore_index = True)
df.head()

# Desriptive Statistics
print(df.describe())

# Split the data - Drop unneccesary columns (non-numerical)
# set 'target' column to its own variable to use later
target = df.target
df.drop(columns=['target'], axis=1, inplace=True)
df.drop(columns=['Machining_Process'], axis=1, inplace=True)

# Inspect - should have 47 columns now
print(df)

# Normalize the data
scaler = sklearn.preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(df)

# Split data using sklearn built in function
X_train, X_test, Y_train, Y_test = train_test_split(scaled_df,target, test_size= 0.2, random_state=42, stratify=target)

# Create global functions 
def dot_product(a, b):
  """
  Purpose: caclulate the dot product of two vectors.
  Params: a, b where 'a' and 'b' are vectors.
  Returns:  dot product of the two vectors.
  """
  return np.matmul(a,b.T)

def sigmoid(x):
  """
  Purpose: Calculate the sigmoid of x to be used as an activation layer function.
  Parameters: x, where x is an array-like/scalar object.
  Returns: The sigmoid function output, which is in the range (0, 1). The output will have the same shape as the input `x`.
  """
  return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
  """
  Purpose: Compute the derivative of the sigmoid function.
    
  Description: The derivative of the sigmoid function shows how the gradient of the sigmoid
  function changes with respect to its input `x`. This is crucial in the 
  backpropagation algorithm of neural networks, where it is used to calculate
  the gradient of the loss function with respect to the weights.
    
  Parameters:
    x (numeric or array-like): Input value or values for which the derivative
    of the sigmoid function is computed. Can be a single number, a list, or
    a NumPy array.
        
  Returns: The derivative of the sigmoid function evaluated at `x`.
  """
  return sigmoid(x) * (1 - sigmoid(x))

def initialize_parameters(X, target):
    """
    Initialize weight(s) as random value(s) and bias(es) as zero.
    """
    shape_X = X.shape[0]
    shape_Y = target.shape[0]

    # initialize parameters that correspond to the shape of the input and output vectors
    W = np.random.randn(shape_Y, shape_X)
    b = np.zeros((shape_Y, 1))

    parameters = {"W": W,
                  "b": b}

    return parameters

def forward_propagation(X, parameters):
    """
    Compute the output of the model.
    """
    W = parameters["W"]
    b = parameters["b"]
    Z = np.matmul(W, X) + b
    A = sigmoid(Z)
    return A 

def compute_cost(Y, Y_hat):
    """
    Compute the cost function.
    """
    m = len(Y)
    cost = np.sum((Y_hat - Y)**2)/(2*m)
    return cost

def backward_propagation(X, Y, Y_hat):
    """
    Compute the gradients of the weight and bias.
    """
    m = len(Y)
    X = np.transpose(X)
    dw = (1 / m) * np.dot(Y_hat - Y, X)
    db = (1 / m) * np.sum(Y_hat - Y)
    return dw, db

def update_parameters(parameters, dw, db, learning_rate):
    """
    Update the weight and bias using the gradients.
    """
    # gradient descent
    parameters["W"] = parameters["W"] - learning_rate * dw
    parameters["b"] = parameters["b"] - learning_rate * db

    return parameters

def nn_model(X, Y, num_iterations=100, learning_rate = 0.01):
    """
    Returns:
    parameters -- parameters (weight(s) and bias(es)) learnt by the model. They can then be used to make predictions.
    """

    iterations = num_iterations # specify the count of training iterations
    learning_rate = learning_rate # adjust the magnitude of the parameter updates

    # Initialize parameters
    parameters = initialize_parameters(X, Y)

    for i in range(iterations):
      # Forward propagation
      Y_hat = forward_propagation(X, parameters)

      # Compute cost
      cost = compute_cost(Y, Y_hat)

      # Backward propagation
      dw, db = backward_propagation(X, Y, Y_hat)

      # Update parameters
      update_parameters(parameters, dw, db, learning_rate)

      # Print cost every 25 iterations
      if i % 25 == 0:
          print(f"Iteration {i}: cost = {cost}")

    return parameters


# Test functionality
# Ensure Y_train and Y_test are the correct shape for your model
Y_train = Y_train.values.reshape(1, Y_train.shape[0])
Y_test = Y_test.values.reshape(1, Y_test.shape[0])

# Train the model
parameters = nn_model(X_train.T, Y_train, num_iterations=2000, learning_rate=0.0075)

# Predict on training and testing sets
Y_pred_train = forward_propagation(X_train.T, parameters) > 0.5
Y_pred_test = forward_propagation(X_test.T, parameters) > 0.5

# Evaluate the model
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(Y_train.flatten(), Y_pred_train.flatten())
test_accuracy = accuracy_score(Y_test.flatten(), Y_pred_test.flatten())

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

"""
Observations- experiment 1 (learning rate = 0.01)
1. Cost function - There is variability in the cost function output. Usually,
Cost function values decrease each epoch, but my model increases/decreases with no 
pattern. This indicates the model is not peforming well and is not accurate
Furthmore, the cost function values are really large (> 4000). Lastly, due to the high
values of the cost function, we can infer that the mdoel is getting its predictions
wrong quit often.

2. Accuracy - Training accuracy = 52.1% and test accuracy 51.8%

Next Steps
- Adjust learning rate
- Create a different activation function (Possibly)
"""