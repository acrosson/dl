
import torch
import torch.nn as nn

class SpatSoftArgMax(nn.Module):
    """Spatial Soft ArgMax Layer for pytorch as used in 
    https://arxiv.org/pdf/1504.00702.pdf aka Spatial Softmax

    Inspired by:
    https://github.com/tensorflow/tensorflow/issues/6271#issuecomment-266893850

    """
    def __init__(self, h, w):
        """
        inputs:
            h   (int)   height of input
            w   (int)   width of input
        """ 
        super(SpatSoftArgMax, self).__init__()
        self.sm2d = nn.Softmax2d()
        self.image_coords = nn.Parameter(torch.zeros((1, h, w, 2)))
        
    def forward(self, x):
        """
        inputs:
            x   (torch.Tensor)  input tensor of shape [N, C, H, W]
        returns:
            x   (torch.Tensor)  output tensor of shape [N, C, 2]
        """
        x = self.sm2d(x)
        x = x.unsqueeze(-1)
        x = self.image_coords * x
        x = x.sum(2).sum(2)
        return x


