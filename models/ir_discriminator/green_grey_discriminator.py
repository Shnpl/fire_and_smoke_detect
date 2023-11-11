import torch
import torch.nn.functional as F
from torch import Tensor
class GreenGreyDiscriminator():
    def __init__(self,grey_thres = 0.01,green_thres = 0.9) -> None:
        self.grey_thres = grey_thres
        self.green_thres = green_thres
    def forward(self, x:Tensor):
        # Get Histogram for each channel
        result = torch.zeros(x.shape[0])
        i = 0
        for image in x:
            r_hist, g_hist, b_hist = self._get_histogram(image)
            mse = F.mse_loss(r_hist, g_hist)+F.mse_loss(g_hist, b_hist)+F.mse_loss(b_hist, r_hist)
            # Gray image usually has similar histogram for each channel
            if mse < self.grey_thres:
                result[i] = 3
            # Green IR image usually has nothing in their red&blue channel, so the histogram usually has its peak at the beginning
            elif torch.sum(r_hist[:16]) > self.green_thres and torch.sum(b_hist[:16]) > self.green_thres:
                    result[i] = 2
            else:
                result[i] = 0
            i+=1
        return result
    def _get_histogram(self, x:Tensor):
        # CxHxW
        r_hist = torch.histc(x[0,:,:], bins=256, min=0, max=1)
        g_hist = torch.histc(x[1,:,:], bins=256, min=0, max=1)
        b_hist = torch.histc(x[2,:,:], bins=256, min=0, max=1)
        # Get histogram for each channel
        return r_hist, g_hist, b_hist
        