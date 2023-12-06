import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class IRDiscriminatorDataset(Dataset):
    def __init__(self,stage= 'train',target_size = 448,enable_greengrey = False) -> None:
        super().__init__()
        self.enable_greengrey = enable_greengrey
        self.stage = stage
        if self.stage == 'train':
            self.root_path = 'datasets/fire_and_smoke_detect/ir_dataset/train'
        elif self.stage == 'val':
            self.root_path = 'datasets/fire_and_smoke_detect/ir_dataset/val'
        elif self.stage == 'test':
            self.root_path = 'datasets/fire_and_smoke_detect/ir_dataset/test'
        else:
            raise ValueError('stage must be train, val or test')

        self.preprocess = transforms.Compose([
            transforms.Resize((target_size,target_size)),
            transforms.ToTensor()
        ])

        self.item_list =[]
        images_list = os.listdir(os.path.join(self.root_path,'images'))
        for image in images_list:
            item = {}
            item['image_path'] = os.path.join(self.root_path,'images',image)
            label_name = "".join(image.split('.')[:-1])+'.txt'
            with open(os.path.join(self.root_path,'labels',label_name),'r') as f:

                item['label'] = f.read()
                if not self.enable_greengrey:
                    if item['label'] == '2' or item['label'] == '3':
                        continue
                    else:
                        self.item_list.append(item)
                else:
                    self.item_list.append(item)
        
        
        self.len = len(self.item_list)
        print(f'Dataset size:{self.len}')
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        item =  self.item_list[index]
        image_path = item['image_path']
        image = Image.open(image_path).convert('RGB')

        image = self.preprocess(image)
        label = int(item['label'])
        
        return image,label
