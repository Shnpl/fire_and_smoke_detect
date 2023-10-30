# train the ir discriminator

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from models.ir_discriminator.ir_discriminator import IRDiscriminator

def train_ir_discriminator():
    model = IRDiscriminator()
    model.train()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    transform = transforms.Compose([
        transforms.Resize((448,448)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder('datasets/ir_discriminator',transform=transform)
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True)
    for epoch in range(10):
        for images,labels in tqdm(dataloader):
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
        print(f'epoch:{epoch},loss:{loss.item()}')
    torch.save(model.state_dict(),f'checkpoints/ir_discriminator/{epoch}.pth')
