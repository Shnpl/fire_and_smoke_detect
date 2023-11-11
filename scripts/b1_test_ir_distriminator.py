# train the ir discriminator
import sys
sys.path.append('.')
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from modules.data.ir_discriminator_dataset import IRDiscriminatorDataset
from models.ir_discriminator.ir_discriminator import IRDiscriminator

def test_ir_discriminator():
    model = IRDiscriminator()
    # load state dict
    model.load_state_dict(torch.load('2.pth'))
    model.eval()
    model.cuda()
    dataset = IRDiscriminatorDataset(stage='test')
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True)
    gt = []
    pred = []
    for images,labels in tqdm(dataloader):
        images = images.cuda()
        outputs = model(images)
        outputs = torch.argmax(outputs,dim=1)
        gt.extend(labels.tolist())
        outputs = outputs.detach().cpu()
        pred.extend(outputs.tolist())
    # calc acc
    correct = 0
    for i in range(len(gt)):
        if gt[i] == pred[i]:
            correct += 1
    acc = correct / len(gt)
    print(f'acc:{acc}')
    #torch.save(model.state_dict(),f'checkpoints/ir_discriminator/{epoch}.pth')
if __name__ == '__main__':
    test_ir_discriminator()