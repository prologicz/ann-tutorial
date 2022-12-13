import numpy as np

# DOT PRODUCTS
# input_vector = [1.72, 1.23]
# weights_1 = [1.26, 0]
# weights_2 = [2.17, 0.32]

# first_indexes_mult = input_vector[0] * weights_1[0]
# second_indexes_mult = input_vector[1] * weights_1[1]
# dot_product_1 = first_indexes_mult + second_indexes_mult

# print (f'The dot product is: {dot_product_1}')

# dot_product_1 = np.dot(input_vector, weights_1)
# print (f'The dot product is: {dot_product_1}')

# MANUAL MODEL
input_vector_correct_pred = np.array([1.66, 1.56])
input_vector_incorrect_pred = np.array([2, 1.5])
weights_1 = np.array([1.45, -.66])
bias = np.array([0.0])

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def make_prediction (input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid (layer_1)
    return layer_2

prediction = make_prediction(input_vector_incorrect_pred, weights_1, bias)
print(f'Prediction: {prediction}')

# LOSS FUNCTION
target = 0
mse = np.square(prediction - target)
print(f'Loss: {mse}')

# OPTIMIZER e.g GRADIENT/DERIVATIVE OF LOSS FUNCTION
derivative = 2 * (prediction - target)
print(f'Derivative: {derivative}')

# RUN 2
weights_1 = weights_1 - derivative
prediction = make_prediction(input_vector_incorrect_pred, weights_1, bias)
error = (target - prediction) ** 2
print(f'New Weights: {weights_1}, Prediction: {prediction}, Error: {error}')

# BACKPROPOGATION

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

layer_1 = np.dot(input_vector_incorrect_pred, weights_1) + bias

derror_dprediction = 2 * (prediction - target)
dprediction_dlayer_1 = sigmoid_deriv(layer_1) 
dlayer1_dbias = 1

derror_dbias = (derror_dprediction * dprediction_dlayer_1 * dlayer1_dbias)
print(derror_dbias)