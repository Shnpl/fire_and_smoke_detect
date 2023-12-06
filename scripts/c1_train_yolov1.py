import sys
sys.path.append('.')
import os
import json

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from models.yolo_v1.yolov1 import Yolov1_Litmodel


if __name__ == '__main__':
    hparams = {
        'lr': 1e-5,
        'batch_size': 8,
        'max_epoch': 64,
        'type':'fire',
        'bands': 'rgb',
        'ir_discriminator':'lightning_logs/ir_discriminator/version_0/checkpoints/epoch=2.ckpt'
    }
    logger = TensorBoardLogger('lightning_logs/',name='yolov1_fire')
    model = Yolov1_Litmodel(hparams)
    trainer = pl.Trainer(devices=1, max_epochs=hparams['max_epoch'], logger=logger)
    trainer.fit(model)