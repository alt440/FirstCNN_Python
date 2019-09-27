import numpy as np
import math

class Softmax:
  # A standard fully-connected layer with softmax activation.

  def __init__(self, input_len, nodes):
    # We divide by input_len to reduce the variance of our initial values
    self.weights = np.random.randn(input_len, nodes) / input_len
    self.biases = np.zeros(nodes)

  def forward(self, max_pool_result_matrix):
      """
      This method will forward us to the next step in this machine (which is getting the final results).
      :param max_pool_result_matrix: The results from my max pooling layer
      :return: The soft max results.
      """
      # Save the shape of the matrix before doing anything
      self.last_input_shape = max_pool_result_matrix.shape

      # First step, take your 3D matrix and flatten it out into a 1D array
      flattened_matrix = max_pool_result_matrix.flatten()

      # cache the result of the matrix before going into the soft max phase
      self.last_input = flattened_matrix

      # then, calculate the results with your weights and biases
      totals = np.dot(flattened_matrix, self.weights) + self.biases
      self.last_totals = totals

      # then calculate the probability of each element using softmax
      return self.softmaxFunction(totals)

  def backprop(self, gradient_loss_expected_result, learning_rate):
        """
        This method back propagates through the softmax layer.
        :param gradient_loss_expected_result: This is the gradient loss for the right answer. (Is it a 1,2,3,4,5,6,7,8, or 9?
        Give the soft max gradient value for that corresponding answer)
        :param expected_result_index: This is the index of the right answer.
        :return: Gradient loss matrix for lower layer.
        """
        expected_result_index = None
        for i in range(gradient_loss_expected_result.shape[0]):
            if gradient_loss_expected_result[i] != 0:
                gradient_loss_expected_result = gradient_loss_expected_result[i]
                expected_result_index = i
                break

        # e^totals
        t_exp = np.exp(self.last_totals)

        # Sum of all e^totals
        S = np.sum(t_exp)

        # Derivatives for softmax function.
        derivative_softmax = -t_exp[expected_result_index] * t_exp / (S ** 2)
        derivative_softmax[expected_result_index] = t_exp[expected_result_index] * (S - t_exp[expected_result_index]) / (S ** 2)

        # Gradients of totals against weights/biases/input
        # derivative of the linear formula t = w * input + b
        # input --> flattened output: partial derivative from input = w, bc (input^1) becomes 1 as we derived.
        # b --> bias from softmax: partial derivative from b = 1
        # w --> weights from softmax: partial derivative from w = input, bc (w^1) becomes 1 as we derived.
        derivative_weights = self.last_input
        derivative_bias = 1
        derivative_inputs = self.weights

        # Gradients of loss against totals
        gradient_loss_matrix = gradient_loss_expected_result * derivative_softmax

        # Gradients of loss against weights/biases/input
        # So now you know that you just need to multiply the gradient loss matrix by all the different derivatives
        gradient_loss_weights = derivative_weights[np.newaxis].T @ gradient_loss_matrix[np.newaxis]
        gradient_loss_bias = gradient_loss_matrix * derivative_bias
        gradient_loss_inputs = derivative_inputs @ gradient_loss_matrix

        # Update weights / biases
        self.weights -= learning_rate * gradient_loss_weights
        self.biases -= learning_rate * gradient_loss_bias

        return gradient_loss_inputs.reshape(self.last_input_shape)


  def softmaxFunction(self, vector):
      """
      Applies the softmax function to give probabilities of the different results.
      :param vector: 1D matrix that is a numpy array
      :return: softmax vector for each of the values
      """
      # for each of the inputs, find their exponential value, and sum them together.
      sum = 0
      for i in range(vector.shape[0]):
          sum += math.exp(vector[i])

      # now determine their probabilities
      answer_vector = np.zeros((vector.shape[0],))
      for i in range(vector.shape[0]):
          answer_vector[i] = (math.exp(vector[i])) / sum

      return answer_vector
