
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class SpatSoftArgMax(nn.Module):

    """Spatial Soft ArgMax Layer for pytorch as used in 
    https://arxiv.org/pdf/1504.00702.pdf aka Spatial Softmax

    Inspired by:
    https://github.com/tensorflow/tensorflow/issues/6271#issuecomment-266893850

    """

    def __init__(self):
        """
        inputs:
            h   (int)   height of input
            w   (int)   width of input
        """ 
        super(SpatSoftArgMax, self).__init__()
        self.sm = MySoftmax()
        
    def forward(self, x):
        """
        inputs:
            x   (torch.Tensor)  input tensor of shape [N, C, H, W]
        returns:
            x   (torch.Tensor)  output tensor of shape [N, C, 2]
        """
        batch_size, num_fp, num_rows, num_cols = x.size()
        num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
        x_map = np.empty([num_rows, num_cols], np.float32)
        y_map = np.empty([num_rows, num_cols], np.float32)

        for i in range(num_rows):
            for j in range(num_cols):
                x_map[i, j] = (i - num_rows / 2.0) / num_rows
                y_map[i, j] = (j - num_cols / 2.0) / num_cols

                # alternative scale map
                #x_map[i, j] = j
                #y_map[i, j] = i

        x_map = Variable(torch.from_numpy(x_map)).cuda()
        y_map = Variable(torch.from_numpy(y_map)).cuda()

        x_map = x_map.view(-1, num_rows * num_cols)
        y_map = y_map.view(-1, num_rows * num_cols)

        features_ = x.view(batch_size, num_fp, num_rows * num_cols)
        softmax = self.sm(features_)

        fp_x = (x_map * softmax).sum(2)
        fp_y = (y_map * softmax).sum(2)

        fp = torch.stack((fp_x, fp_y), 2)
        return fp

class MySoftmax(nn.Module):
    def forward(self, input_):
        batch_size = input_.size()[0]
        output_ = torch.stack([F.softmax(input_[i]) for i in range(batch_size)], 0)
        return output_

