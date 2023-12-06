import sys
sys.path.append('.')
import os
import json

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from models.yolo_v1.yolov1 import Yolov1_Litmodel


if __name__ == '__main__':
    logger = TensorBoardLogger('lightning_logs/',name='yolov1_fire')
    model = Yolov1_Litmodel.load_from_checkpoint('lightning_logs/yolov1_fire/version_0/checkpoints/epoch=15-step=4416.ckpt')
    trainer = pl.Trainer(devices=1, logger=logger)
    trainer.validate(model)