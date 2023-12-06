from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS, OptimizerLRScheduler
import torch
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from modules.data.ir_discriminator_dataset import IRDiscriminatorDataset
from torchvision.utils import save_image

class GreenGreyDiscriminator():
    def __init__(self,grey_thres = 1e-12,green_thres = 0.99) -> None:
        self.grey_thres = grey_thres
        self.green_thres = green_thres
    @ torch.no_grad()
    def forward(self, x:Tensor):
        # Get Histogram for each channel
        result = torch.zeros(x.shape[0])
        x_in = x.clone().cpu().detach()
        i = 0
        for image in x_in:
            r_hist, g_hist, b_hist = self._get_histogram(image)
            # hist normalization
            total = x_in.shape[-2]*x_in.shape[-1]
            r_hist = r_hist/total
            g_hist = g_hist/total
            b_hist = b_hist/total
            mse = F.mse_loss(r_hist, g_hist)+F.mse_loss(g_hist, b_hist)+F.mse_loss(b_hist, r_hist)
            # Gray image usually has similar histogram for each channel
            #save_image(image,'test.png')
            if mse < self.grey_thres:
                result[i] = 3
            # Green IR image usually has nothing in their red&blue channel, so the histogram usually has its peak at the beginning
            elif torch.sum(r_hist[:64]) > self.green_thres and torch.sum(b_hist[:64]) > self.green_thres:
                    result[i] = 2
            else:
                result[i] = 0
            i+=1
        return result
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    def _get_histogram(self, x:Tensor):
        # CxHxW
        r_hist = torch.histc(x[0,:,:], bins=256, min=0, max=1)
        g_hist = torch.histc(x[1,:,:], bins=256, min=0, max=1)
        b_hist = torch.histc(x[2,:,:], bins=256, min=0, max=1)
        # Get histogram for each channel
        return r_hist, g_hist, b_hist

class GreenGreyDiscriminator_Litmodel(pl.LightningModule):
    def __init__(self,hparams:dict) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = GreenGreyDiscriminator()
        self.train_dataset = IRDiscriminatorDataset(stage='train',enable_greengrey=True)
        self.val_dataset = IRDiscriminatorDataset(stage='val',enable_greengrey=True)
        self.test_dataset = IRDiscriminatorDataset(stage='test',enable_greengrey=True)
    def on_validation_epoch_start(self) -> None:
        self.val_correct_num = 0
        self.val_total_num = 0
        self.green_num = 0
        self.grey_num = 0
        self.green_correct_num = 0
        self.grey_correct_num = 0
        self.rgb_as_green_num = 0
        self.rgb_as_grey_num = 0
    def forward(self,x) :
        return self.model(x)
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        # the green grey discriminator only cares about green and grey images, so red&rgb -> 0
        # 0->0,1->0,2->2,3->3
        new_labels = torch.zeros(labels.shape,device=self.device,dtype=torch.int64)
        new_labels = torch.where(labels == 2,torch.tensor(2),new_labels)
        new_labels = torch.where(labels == 3,torch.tensor(3),new_labels)
        labels = new_labels
        # if there're green images:


        output = self.model(images)
        for i in range(len(labels)):
            pred = output[i]
            gt = labels[i]
            if gt == 2:
                self.green_num += 1
                if pred == 2:
                    self.green_correct_num += 1
            elif gt == 3:
                self.grey_num += 1
                if pred == 3:
                    self.grey_correct_num += 1
            if gt == pred:
                self.val_correct_num += 1
            if gt == 0 and pred == 2:
                self.rgb_as_green_num += 1
            if gt == 0 and pred == 3:
                self.rgb_as_grey_num += 1
            self.val_total_num += 1
    
    def on_validation_epoch_end(self) -> None:
        print(f'val total acc: {self.val_correct_num/self.val_total_num}')
        print(f'green acc: {self.green_correct_num/self.green_num}')
        print(f'grey acc: {self.grey_correct_num/self.grey_num}')
        print(f'green num: {self.green_num},correct num: {self.green_correct_num}')
        print(f'grey num: {self.grey_num}',f'correct num: {self.grey_correct_num}')
        print(f'other num: {self.val_total_num-self.green_num-self.grey_num},correct num: {self.val_correct_num-self.green_correct_num-self.grey_correct_num}')
        print(f'other acc: {(self.val_correct_num-self.green_correct_num-self.grey_correct_num)/(self.val_total_num-self.green_num-self.grey_num)}')
        print(f'rgb as green num: {self.rgb_as_green_num}')
        print(f'rgb as grey num: {self.rgb_as_grey_num}')
    def on_test_epoch_start(self):
        self.test_correct_num = 0
        self.test_total_num = 0
        self.test_green_num = 0
        self.test_grey_num = 0
        self.test_green_correct_num = 0
        self.test_grey_correct_num = 0
        self.test_rgb_as_green_num = 0
        self.test_rgb_as_grey_num = 0
    def test_step(self, batch, batch_idx):
        images, labels = batch
        # the green grey discriminator only cares about green and grey images, so red&rgb -> 0
        # 0->0,1->0,2->2,3->3
        new_labels = torch.zeros(labels.shape,device=self.device,dtype=torch.int64)
        new_labels = torch.where(labels == 2,torch.tensor(2),new_labels)
        new_labels = torch.where(labels == 3,torch.tensor(3),new_labels)
        labels = new_labels
        # if there're green images:


        output = self.model(images)
        for i in range(len(labels)):
            pred = output[i]
            gt = labels[i]
            if gt == 2:
                self.test_green_num += 1
                if pred == 2:
                    self.test_green_correct_num += 1
            elif gt == 3:
                self.test_grey_num += 1
                if pred == 3:
                    self.test_grey_correct_num += 1
            if gt == pred:
                self.test_correct_num += 1
            if gt == 0 and pred == 2:
                self.test_rgb_as_green_num += 1
            if gt == 0 and pred == 3:
                self.test_rgb_as_grey_num += 1
            self.test_total_num += 1  
    def on_test_epoch_end(self):
        print(f'test total  acc: {self.test_correct_num/self.test_total_num}')
        print(f'green acc: {self.test_green_correct_num/self.test_green_num}')
        print(f'grey acc: {self.test_grey_correct_num/self.test_grey_num}')
        print(f'green num: {self.test_green_num},correct num: {self.test_green_correct_num}')
        print(f'grey num: {self.test_grey_num}',f'correct num: {self.test_grey_correct_num}')
        print(f'other num: {self.test_total_num-self.test_green_num-self.test_grey_num},correct num: {self.test_correct_num-self.test_green_correct_num-self.test_grey_correct_num}')
        print(f'other acc: {(self.test_correct_num-self.test_green_correct_num-self.test_grey_correct_num)/(self.test_total_num-self.test_green_num-self.test_grey_num)}')
        print(f'rgb as green num: {self.test_rgb_as_green_num}')
        print(f'rgb as grey num: {self.test_rgb_as_grey_num}')
    def configure_optimizers(self):
        # it's actually not used, since it's not a DL model, but the PL requires it
        return super().configure_optimizers()
    def val_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.hparams['batch_size'],shuffle=True,num_workers=10)
    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.hparams['batch_size'],shuffle=False,num_workers=10)
    
    