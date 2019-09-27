import numpy as np

class MaxPool2:
  # A Max Pooling layer using a pool size of 2.

  def forward(self, feature_matrix):
      """
      This does max pooling. In doing so, the minimum value it takes is 0 (like the ReLU function) to avoid any problem
      related to the activation function using sigmoids.
      :param feature_matrix: the cnn result from the extractFeaturesWithKernel method, which is after having applied many kernel filters
      So this feature_matrix is 3D.
      :return: the smaller matrix that extracts the maximum value of some n * n dimension kernel
      """
      # save the input you received (3D matrix with depth of number of filters you applied)
      self.last_input = feature_matrix

      # size of resulting array will be feature_matrix / size, +1 if result > .0
      size_resultX = feature_matrix.shape[0] / 2
      size_resultY = feature_matrix.shape[1] / 2
      if size_resultX % 1 != 0:
          size_resultX += 1
      if size_resultY % 1 != 0:
          size_resultY += 1

      size_resultX = int(size_resultX)
      size_resultY = int(size_resultY)

      # container of all max_pool results
      max_pool_results = np.zeros((size_resultX, size_resultY, feature_matrix.shape[2]))

      # iterate over the filters I have applied as well.
      for h in range(feature_matrix.shape[2]):

          result_matrix = np.zeros((size_resultX, size_resultY))

          # follows the index on the feature_matrix
          index_i_feature = 0
          index_j_feature = 0

          for i in range(size_resultX):
              for j in range(size_resultY):
                  # extract max value
                  max_val = -2
                  # going through the part of size * size to find max value
                  for k in range(index_i_feature, index_i_feature + 2):
                      for l in range(index_j_feature, index_j_feature + 2):
                          if k < feature_matrix.shape[0] and l < feature_matrix.shape[1] and feature_matrix[
                              k, l, h] > max_val:
                              max_val = feature_matrix[k, l, h]

                  # now save the val
                  result_matrix[i, j] = max_val

                  # increment for j
                  index_j_feature += 2

              # increment for i
              index_i_feature += 2
              index_j_feature = 0

          # now add this result matrix to the list of all results
          max_pool_results[:, :, h] = result_matrix

      return max_pool_results


  def backprop(self, d_L_d_out):
    '''
    Performs a backward pass of the maxpool layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    '''
    d_L_d_input = np.zeros(self.last_input.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      h, w, f = im_region.shape
      amax = np.amax(im_region, axis=(0, 1))

      for i2 in range(h):
        for j2 in range(w):
          for f2 in range(f):
            # If this pixel was the max value, copy the gradient to it.
            if im_region[i2, j2, f2] == amax[f2]:
              d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

    return d_L_d_input
