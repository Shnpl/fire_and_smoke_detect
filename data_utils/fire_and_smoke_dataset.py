import os
from PIL import Image

import torch
import math
from torch.utils.data import Dataset,DataLoader
import torchvision
import torchvision.transforms as transforms

class FireSmokeDataset(Dataset):
    def __init__(self,stage= 'train',type='fire') -> None:
        super().__init__()
        self.stage = stage
        if type == 'fire':
            if self.stage == 'train':
                self.root_path = 'datasets/fire_and_smoke_detect/Fire/train'
            elif self.stage == 'val':
                self.root_path = 'datasets/fire_and_smoke_detect/Fire/val'
            elif self.stage == 'test':
                self.root_path = 'datasets/fire_and_smoke_detect/Fire/test'
            else:
                raise ValueError('stage must be train, val or test')
        elif type == 'smoke':
            if self.stage == 'train':
                self.root_path = 'datasets/fire_and_smoke_detect/Fog/train'
            elif self.stage == 'val':
                self.root_path = 'datasets/fire_and_smoke_detect/Fog/val'
            elif self.stage == 'test':
                self.root_path = 'datasets/fire_and_smoke_detect/Fog/test'
            else:
                raise ValueError('stage must be train, val or test')

        self.preprocess = transforms.Compose([
            transforms.Resize((448,448)),
            transforms.ToTensor()
        ])

        self.item_list =[]
        images_list = os.listdir(os.path.join(self.root_path,'images'))
        for image in images_list:
            item = {}
            item['image_path'] = os.path.join(self.root_path,'images',image)
            label = "".join(image.split('.')[:-1])+'.txt'
            item['label'] = os.path.join(self.root_path,'labels',label)
            self.item_list.append(item)
        
        
        self.len = len(self.item_list)
        print(f'Dataset size:{self.len}')
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        item =  self.item_list[index]
        image_path = item['image_path']
        label_path = item['label']
        
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image)   

        # Label Encoding
		# [{'name': '', 'xmin': '', 'ymin': '', 'xmax': '', 'ymax': '', }, {}, {}, ...]
		# ==>
		# [x, y  (relative to cell), width, height, 1 if exist (confidence),
		#  x, y  (relative to cell), width, height, 1 if exist (confidence),
		#  one-hot encoding of 1 categories]

        label = torch.zeros((7, 7, 11))
        with open(label_path,'r') as f:
            for line in f:
                line = line.replace('\n','').split(' ')

                name= 0
                x_center = float(line[1])
                y_center = float(line[2])
                width = float(line[3])
                height = float(line[4])        

                xmin = x_center - width / 2.0
                ymin = y_center - height / 2.0
                xmax = x_center + width / 2.0
                ymax = y_center + height / 2.0

                if xmin == xmax or ymin == ymax:
                    continue
                if xmin >= 1 or ymin >= 1 or xmax <= 0 or ymax <= 0:
                    continue

                xidx = math.floor(x_center * 7.0)
                yidx = math.floor(y_center * 7.0)
                

                # According to the paper
                # if multiple objects exist in the same cell
                # pick the one with the largest area
                if label[yidx][xidx][4] == 1: # already have object
                    if label[yidx][xidx][2] * label[yidx][xidx][3] < width * height:
                        use_data = True
                    else: use_data = False
                else: use_data = True

                if use_data:
                    for offset in [0, 5]:
                        # Transforming image relative coordinates to cell relative coordinates:
                        # x - idx / 7.0 = x_cell / cell_count (7.0)
                        # => x_cell = x * cell_count - idx = x * 7.0 - idx
                        # y is the same
                        label[yidx][xidx][0 + offset] = x_center * 7.0 - xidx
                        label[yidx][xidx][1 + offset] = y_center * 7.0 - yidx
                        label[yidx][xidx][2 + offset] = width
                        label[yidx][xidx][3 + offset] = height
                        label[yidx][xidx][4 + offset] = 1
                    label[yidx][xidx][10 + name] = 1
            return image, label

if __name__ == '__main__':
    ds = FireSmokeDataset('val')
    for image,label in ds:
        print(label)
