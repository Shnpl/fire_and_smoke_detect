from PIL import Image
import os
from matplotlib import pyplot as plt
from matplotlib import colormaps
import numpy as np
class ir_temp_getter():
    def __init__(self,grayscale_level=256) -> None:
        self.grayscale_level = grayscale_level
        self.cmap = colormaps.get_cmap('plasma')
        self.idxs = [i for i in range(self.grayscale_level)]
        self.vals = [i/self.grayscale_level for i in self.idxs]
        self.colors = self.cmap(self.vals)
        self.colors = np.array(self.colors[:,0:3],dtype=np.float32)
    def get_temperature(self,img_mat):
        #img_mat: CHW
        _,h,w = img_mat.shape 
        R = img_mat[0,:,:]
        G = img_mat[1,:,:]
        B = img_mat[2,:,:]
        distance = np.zeros((self.grayscale_level,h,w))
        R = np.stack([R]*self.grayscale_level)
        G = np.stack([G]*self.grayscale_level)
        B = np.stack([B]*self.grayscale_level)

        R = (R-self.colors[:,0].reshape(self.grayscale_level,1,1))**2
        G = (G-self.colors[:,1].reshape(self.grayscale_level,1,1))**2
        B = (B-self.colors[:,2].reshape(self.grayscale_level,1,1))**2
        # for i in range(256):
        #     distance[i,:,:] = (R-self.colors[i,0])**2 + (G-self.colors[i,1])**2 + (B-self.colors[i,2])**2
        distance = R+G+B    
        idx =np.argmin(distance,axis=0)
        # to float
        idx = idx.astype(np.float32)
        
        idx /= self.grayscale_level
        return idx