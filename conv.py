import numpy as np

'''
Note: In this implementation, we assume the input is a 2d numpy array for simplicity, because that's
how our MNIST images are stored. This works for us because we use it as the first layer in our
network, but most CNNs have many more Conv layers. If we were building a bigger network that needed
to use Conv3x3 multiple times, we'd have to make the input be a 3d numpy array.
'''

class Conv3x3:
  # A Convolution layer using 3x3 filters.

  def __init__(self, num_filters):
    self.num_filters = num_filters

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3) / 9

  def forward(self, image_matrix):
      """
      Applies a forward pass (Passes through convolution layer). We do convolutions of the images with our kernels
      :param image_matrix: The image that we want to analyze
      :return: Matrices that represent the image from different filters' point of view shape = (image.width, image.height, nb_filters)
      """
      # before starting, set the input that was before the convolution
      self.last_input = image_matrix
      # this is the result of all the convolutions and all the filters
      result_convolutions = np.zeros((image_matrix.shape[0], image_matrix.shape[1], self.num_filters))
      # iteration count
      iter = 0
      # getting max value to normalize
      max_val = 0

      for kernel in self.filters:
          # finding the center of kernel to avoid coordinates not defined for the values going outside of the image.
          # We want the center point of the kernel or filter to start at (0,0), thus the kernel point left of center is not
          # defined. NOTE : I start at (0,0) to have filter results of the same size as the image!
          shape_kernel = kernel.shape
          # find center point of the shape.
          center = shape_kernel[0] // 2

          # resulting matrix
          feature_matrix = np.zeros(image_matrix.shape)
          # result matrix from kernel operations
          result_matrix_sum = 0

          # now go over image_matrix using kernel
          for i in range(image_matrix.shape[0]):
              for j in range(image_matrix.shape[1]):
                  # apply kernel
                  init_row = -center
                  init_col = -center
                  for k in range(shape_kernel[0]):
                      for l in range(shape_kernel[1]):
                          # if the location is out of bounds for the image_matrix
                          if i + init_row < 0 or j + init_col < 0 or i + init_row >= image_matrix.shape[0] or \
                                  j + init_col >= image_matrix.shape[1]:
                              result_matrix_sum += 0
                          else:
                              result_matrix_sum += kernel[k, l] * image_matrix[i + init_row, j + init_col]
                          init_col += 1
                      init_row += 1
                      init_col = -center

                  # now I have my result_matrix_sum. I put it to the same location on my other matrix as the current index
                  # in image_matrix
                  feature_matrix[i, j] = result_matrix_sum
                  # see if this value is the max
                  if result_matrix_sum > max_val:
                      max_val = result_matrix_sum
                  result_matrix_sum = 0

          # now store that matrix in our result matrix
          result_convolutions[:, :, iter] = feature_matrix
          # increase interation count
          iter += 1

      # now normalize the data. Remember you need to back propagate that part too! Otherwise big change in results
      # result_convolutions /= max_val
      # self.max_value = max_val
      # val = self.forward_(image_matrix)
      return result_convolutions


  def backprop(self, gradient_matrix_max_pool, learn_rate):
      """
      Performs the backward propagation for this layer.
      :param gradient_matrix_max_pool: Loss gradients from the layer on top, which is the max pool layer.
      :return: None. This is the last step in the back propagation, because this was the first step of our CNN.
      """
      # last thing I do in forward, so must be first thing I do in back prop.
      # gradient_matrix_max_pool*= self.max_value
      gradient_filters = np.zeros(self.filters.shape)

      # index kernel
      index_kernel = 0
      # so here we redo what was done in the forward pass. However, here we
      # put our values into the matrix gradient_filters. We do some inverse convolution, where
      # instead of imprinting values inside the image, we imprint inside gradient_filters.
      # gradient_filters[k] += gradient_matrix_max_pool[i,j,k] * input_image_region
      # remember: this is because a convolution is the result of multiplying a region.
      # now go over image_matrix using kernel
      for kernel in gradient_filters:
          # finding the center of kernel to avoid coordinates not defined for the values going outside of the image.
          # We want the center point of the kernel or filter to start at (0,0), thus the kernel point left of center is not
          # defined. NOTE : I start at (0,0) to have filter results of the same size as the image!
          shape_kernel = kernel.shape
          # find center point of the shape.
          center = shape_kernel[0] // 2

          for i in range(gradient_matrix_max_pool.shape[0]):
              for j in range(gradient_matrix_max_pool.shape[1]):
                  # apply kernel
                  init_row = -center
                  init_col = -center
                  for k in range(shape_kernel[0]):
                      for l in range(shape_kernel[1]):
                          # if the location is out of bounds for the gradient_matrix_max_pool. I can do this because
                          # gradient_matrix_max_pool has the same dimensions as the self.last_input
                          if i + init_row < 0 or j + init_col < 0 or i + init_row >= gradient_matrix_max_pool.shape[
                              0] or \
                                  j + init_col >= gradient_matrix_max_pool.shape[1]:
                              # do it from kernel's view
                              gradient_filters[index_kernel, k, l] += 0
                          else:
                              gradient_filters[index_kernel, k, l] += gradient_matrix_max_pool[
                                                                          i + init_row, j + init_col, index_kernel] \
                                                                      * (self.last_input[i + init_row, j + init_col])

                          init_col += 1
                      init_row += 1
                      init_col = -center

          index_kernel += 1

      self.filters -= learn_rate * gradient_filters

      return self.filters
