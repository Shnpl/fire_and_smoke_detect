# A simple discriminator, which can classify the input as ir image or RGB image.

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch import nn

from modules.data.ir_discriminator_dataset import IRDiscriminatorDataset

class IRDiscriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024*7*7, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class IRDiscriminator_Litmodel(pl.LightningModule):
    def __init__(self,hparams:dict) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = IRDiscriminator()
        
        if self.hparams['loss'] == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f'loss {self.hparams["loss"]} not implemented')
        
        self.train_dataset = IRDiscriminatorDataset(stage='train')
        self.val_dataset = IRDiscriminatorDataset(stage='val')
        self.test_dataset = IRDiscriminatorDataset(stage='test')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        # labels to one-hot
        labels = torch.zeros(labels.shape[0],2).to(self.device).scatter_(1,labels.unsqueeze(1),1,)

        pred = self.model(images)
        loss = self.loss(pred,labels)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        # labels to one-hot
        labels = torch.zeros(labels.shape[0],2).to(self.device).scatter_(1,labels.unsqueeze(1),1,)

        pred = self.model(images)
        loss = self.loss(pred,labels)
        self.log('val_loss', loss.item())
        return loss
    def on_test_epoch_start(self) -> None:
        print("test epoch start")
        self.correct = 0
        self.total = 0

    def test_step(self, batch, batch_idx):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        pred = self.model(images)

        pred = torch.argmax(pred,dim=1)
        # calc acc
        for i in range(len(labels)):
            if labels[i] == pred[i]:
                self.correct += 1
            self.total += 1

        print("------------------")
        print(f"pred:{pred}")
        print(f"labels:{labels}")
        print("------------------")
    def on_test_epoch_end(self) -> None:
        acc = self.correct / self.total
        print(f"acc:{acc},total:{self.total},wrong:{self.total-self.correct}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return optimizer
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams['batch_size'], num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams['batch_size'], num_workers=10)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams['batch_size'], num_workers=10)       

if __name__ == '__main__':
    model = IRDiscriminator()
    x = torch.randn(1,3,448,448)
    y = model(x)
    print(y.shape)