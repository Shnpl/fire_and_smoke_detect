import sys
sys.path.append('.')
import os
import torch
from models.Yet_Another_YOLOv4_Pytorch.pl_model import YOLOv4PL
from lightning.pytorch.utilities.parsing import AttributeDict
import pytorch_lightning as pl
from argparse import Namespace
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

ckpt_path = 'lightning_logs/final/fire/2t12n2tq_epoch=73-step=40848_replace_hparams.ckpt'
ckpt = torch.load(ckpt_path)
ckpt['hyper_parameters']['valid_ds'] = 'datasets/fire_and_smoke_detect/Fire/test/test.txt'
torch.save(ckpt, 'temp.ckpt')
m = YOLOv4PL.load_from_checkpoint('temp.ckpt')

t = pl.Trainer(logger = False,
           devices=[1],
           precision=32,
           benchmark=True,
           #auto_lr_find=True


#            resume_from_checkpoint="model_checkpoints/yolov4epoch=82.ckpt",
        #    auto_lr_find=True,
          #  auto_scale_batch_size='binsearch',
        #    fast_dev_run=True
          )
t.validate(m)