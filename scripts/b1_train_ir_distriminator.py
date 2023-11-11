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

def train_ir_discriminator():
    model = IRDiscriminator()
    model.train()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = IRDiscriminatorDataset()
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True)
    for epoch in range(10):
        for images,labels in tqdm(dataloader):
            images = images.cuda()
            labels = labels.cuda()
            # labels to one-hot
            labels = torch.zeros(labels.shape[0],2).cuda().scatter_(1,labels.unsqueeze(1),1,)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
        print(f'epoch:{epoch},loss:{loss.item()}')
    #torch.save(model.state_dict(),f'checkpoints/ir_discriminator/{epoch}.pth')
if __name__ == '__main__':
    train_ir_discriminator()