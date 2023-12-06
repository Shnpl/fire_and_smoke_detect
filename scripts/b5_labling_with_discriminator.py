import sys
sys.path.append('.')
import os
import yaml
import tqdm
import torch
from torchvision.io import read_image
import torch.nn.functional as F
from torchvision import transforms
from models.ir_discriminator.green_grey_discriminator import GreenGreyDiscriminator_Litmodel
from models.ir_discriminator.ir_discriminator import IRDiscriminator_Litmodel
from PIL import Image
target_size = 448
preprocess = transforms.Compose([
            transforms.Resize((target_size,target_size)),
            transforms.ToTensor()
        ])
hparams = {
    'batch_size':1,
}
greengrey_model = GreenGreyDiscriminator_Litmodel(hparams)
dir = 'datasets/fire_and_smoke_detect/Fire/test/images'
band_label_path = 'datasets/fire_and_smoke_detect/Fire/test/band.txt'
if not os.path.exists(band_label_path):
    print('band.txt not found')
image_path_list = os.listdir(dir)
image_path_list.sort()
image_path_list = [os.path.join(dir, x) for x in image_path_list]

# load ir discriminator
path = 'lightning_logs/ir_discriminator/version_0'
epoch = 2
with open(f'{path}/hparams.yaml') as f:
    hparams = yaml.load(f,Loader=yaml.FullLoader)
    ir_model = IRDiscriminator_Litmodel.load_from_checkpoint(f'{path}/checkpoints/epoch={epoch}.ckpt')
    ir_model.eval()
    ir_model.freeze()
    ir_model.cpu()




with open(band_label_path,'w') as f:
    with tqdm.tqdm(total=len(image_path_list)) as pbar:
        for image_path in image_path_list:
            image = Image.open(image_path).convert('RGB')
            image = preprocess(image)
            image = image.unsqueeze(0)# C,H,W -> B,C,H,W
            pred = greengrey_model(image)
            if pred == 0:
                pred_ir = ir_model(image)
                pred_ir = F.softmax(pred_ir,dim=1)
                pred_ir = int(torch.argmax(pred_ir,dim=1))
                if pred_ir == 0:
                    result = 0
                else :
                    result = 1
            elif pred == 2:
                result = 2
            elif pred == 3:
                result = 3
            f.write(f'{image_path} {result}\n')
            pbar.update(1)