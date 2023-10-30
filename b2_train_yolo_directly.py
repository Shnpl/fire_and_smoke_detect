import os
import json
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torchvision
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.utils import make_grid,save_image
from torchvision.models.resnet import ResNet50_Weights
from models.yolo_v1.yolo import Yolo,yolo_loss,nms
from models.yolo_v1.utils.visualize import draw_ground_truth,draw_detection_result,tensor_to_cv2,cv2_to_PIL
from data_utils.fire_and_smoke_dataset import FireSmokeDataset
from torch.utils.data import DataLoader

class YoloDirectLitmodel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        resnet50 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone = nn.Sequential(*list(resnet50.children())[:-2]) # remove avg pool and fc
        self.model = Yolo(backbone, backbone_out_channels=2048)
        self.loss = yolo_loss

        self.train_dataset = FireSmokeDataset(stage='train')
        self.val_dataset = FireSmokeDataset(stage='val')
        self.test_dataset = FireSmokeDataset(stage='test')

    def forward(self, x):
        return self.model(x)
    def on_train_epoch_start(self) -> None:
        self.train_loss =[]
    def training_step(self, batch, batch_idx):
        images,labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        yhat = self.forward(images)
        loss = yolo_loss(yhat, labels)
        loss = loss.mean()
        self.train_loss.append(loss.item())
        return loss
    def on_train_epoch_end(self) -> None:
        train_loss = torch.mean(torch.tensor(self.train_loss))
        print(f'Epoch {self.current_epoch} train loss: {train_loss}')
    def on_validation_epoch_start(self) -> None:
        self.save_i = 0
        self.val_loss = []
    def validation_step(self, batch, batch_idx):
        # val loss
        images,labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
       
        yhats = self.forward(images)
        loss = yolo_loss(yhats, labels)
        loss = loss.mean()
        self.val_loss.append(loss.item())
        # visualize
        length = len(images)

        yhats = yhats.reshape(-1,7,7,11)
        save_dir = self.logger.log_dir
        if not os.path.exists(os.path.join(save_dir, 'test')):
            os.makedirs(os.path.join(save_dir, 'test'))
        for i in range(length):
            yhat = nms(yhats[i])
            image = images[i]
            label = labels[i]
            pred = draw_detection_result(tensor_to_cv2(image),yhat, raw=False, thres=0.3)
            # label: 7x7x11
            tmp = []
            for yidx in range(7):
                for xidx in range(7):
                    cell = label[yidx][xidx]
                    if cell[4] == 1:

                        x_center = (float(cell[0])+xidx)/7.0
                        y_center = (float(cell[1])+yidx)/7.0
                        width = float(cell[2])
                        height = float(cell[3])

                        xmin = x_center - width / 2.0
                        ymin = y_center - height / 2.0
                        xmax = x_center + width / 2.0
                        ymax = y_center + height / 2.0



                        tmp_dict = {
                            'xmin': xmin,
                            'ymin': ymin,
                            'xmax': xmax,
                            'ymax': ymax,
                            'category': 0
                        }
                        tmp.append(tmp_dict)
            tmp_json = json.dumps(tmp)
        
            pred = cv2_to_PIL(draw_ground_truth(pred,tmp_json))
            pred = torchvision.transforms.ToTensor()(pred)
            save_path = os.path.join(save_dir, 'test', f'epoch{self.current_epoch}_result{self.save_i}.png')
            self.save_i += 1
            save_image(pred, save_path)
            break
        
        return loss
    def on_validation_epoch_end(self) -> None:
        val_loss = torch.mean(torch.tensor(self.val_loss))
        print(f'Epoch {self.current_epoch} val loss: {val_loss}')
    def test_step(self,batch, batch_idx) -> STEP_OUTPUT:
        pass


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=10)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.hparams.batch_size, num_workers=10)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.forward(x)
        return y_hat

    def predict_epoch_end(self, outputs):
        return outputs

    def predict(self, dataset):
        self.predict_dataset = dataset
        return self.trainer.predict(self)

if __name__ == '__main__':
    hparams = {
        'lr': 1e-5,
        'batch_size': 8,
    }
    model = YoloDirectLitmodel(hparams)
    trainer = pl.Trainer(devices=1, max_epochs=64)
    trainer.fit(model)