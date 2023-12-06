import sys
sys.path.append('.')
from models.Yet_Another_YOLOv4_Pytorch.pl_model import YOLOv4PL
from lightning.pytorch.utilities.parsing import AttributeDict
import pytorch_lightning as pl
from argparse import Namespace
import torch
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint,ModelSummary
from pytorch_lightning.loggers import WandbLogger

# hparams = {
#     "n_classes" : 2,
#     "weights_path": "lightning_logs/yolov4/yolov4_pretrain.pth",
#     "pretrained" : True,
#     "train_ds" : "datasets/fire_and_smoke_detect/Fog/train/train.txt",
#     "valid_ds" : "datasets/fire_and_smoke_detect/Fog/val/val.txt",
#     "img_extensions" : [".JPG", ".jpg"],
#     "bs" : 4,
#     "momentum": 0.9,
#     "wd": 0.001,
#     "lr": 1e-4,
#     "epochs" : 100,
#     "pct_start" : 10/100,
    
#     "optimizer" : "Adam",
#     "flat_epochs" : 50,
#     "cosine_epochs" : 25,
#     "scheduler" : "Cosine Delayed", 
    
#     "SAT" : False,
#     "epsilon" : 0.1,
#     "SAM" : False,
#     "ECA" : False,
#     "WS" : False,
#     "Dropblock" : False,
#     "iou_aware" : False,
#     "coord" : False,
#     "hard_mish" : False,
#     "asff" : False,
#     "repulsion_loss" : False,
#     "acff" : True,
#     "bcn" : False,
#     "mbn" : False,
# }
# Original
# hparams = {
#     "n_classes" : 1,
#     "weights_path": "lightning_logs/yolov4/yolov4_pretrain.pth",
#     "pretrained" : True,
#     "train_ds" : "datasets/fire_and_smoke_detect/Fog/train/train.txt",
#     "valid_ds" : "datasets/fire_and_smoke_detect/Fog/val/val.txt",
#     "img_extensions" : [".JPG", ".jpg"],
#     "bs" : 4,
#     "momentum": 0.9,
#     "wd": 0.001,
#     "lr": 1e-4,
#     "epochs" : 100,
#     "pct_start" : 10/100,
    
#     "optimizer" : "Adam",
#     "flat_epochs" : 50,
#     "cosine_epochs" : 25,
#     "scheduler" : "Cosine Delayed", 
    
#     "SAT" : True,# 结构无关，可以加
#     "epsilon" : 0.1,# SAT的参数
#     "SAM" : False,# 确定
#     "ECA" : False,# 确定
#     "WS" : False,# 确定
#     "Dropblock" : False,# 确定
#     "iou_aware" : False,# 确定
#     "coord" : False,# 确定
#     "hard_mish" : False,# 确定
#     "asff" : False,# 确定
#     "repulsion_loss" : False,# 确定
#     "acff" : False,# 确定
#     "bcn" : False,# 确定
#     "mbn" : False,# 确定
# }
hparams = {
    "name":"yolov4_fog_head_fullsize_augmentation_off_mosaic_off_cosine_warmup_sgd",
    "n_classes" : 1,
    "weights_path": "lightning_logs/yolov4/yolov4_pretrain.pth",
    "pretrained" : True,
    "train_ds" : "datasets/fire_and_smoke_detect/detection/Fog/train/train.txt",
    "valid_ds" : "datasets/fire_and_smoke_detect/detection/Fog/val/val.txt",
    "img_extensions" : [".JPG", ".jpg"],
    "bs" : 4,
    "momentum": 0.9,
    "wd": 0.001,
    "lr": 1e-4,
    "epochs" : 50,
    
    
    "optimizer" : "SGD",
    #"flat_epochs" : 50,# Cosine Delayed use it
    #"cosine_epochs" : 25, # Cosine Delayed use it
    "scheduler" : "Cosine Warm-up", 
    "pct_start" : 10/100,
    
    "SAT" : False,# 结构无关，可以加
    "epsilon" : 0.1,# SAT的参数
    "SAM" : False,# 确定
    "ECA" : False,# 确定
    "WS" : False,# 确定
    "Dropblock" : False,# 确定
    "iou_aware" : False,# 确定
    "coord" : False,# 确定
    "hard_mish" : False,# 确定
    "asff" : False,# 确定
    "repulsion_loss" : False,# 确定
    "acff" : False,# 确定
    "bcn" : False,# 确定
    "mbn" : False,# 确定
    "train_part":['backbone','neck','head'],
    "anchors":[[[ 49, 70],[111,  85],[65, 177]],
               [[148, 149],[269, 92],[107, 285]],
               [[205, 260],[366, 178],[375, 396]]],
    "mosaic_on":False,
    "augmentation_on":False,
}
hparams = AttributeDict(hparams)
m = YOLOv4PL(hparams)
wandb_logger = WandbLogger(save_dir='lightning_logs/', 
                           name = hparams.name,
                           project='Fire_and_Smoke_Detection')
checkpoint_callback = ModelCheckpoint(
    every_n_epochs=1,
    save_top_k=-1,
    verbose=True,
)
torch.autograd.set_detect_anomaly(True)
t = pl.Trainer(logger = wandb_logger,
           devices=[3],
           precision=32,
           benchmark=True,
           callbacks=[checkpoint_callback],
           max_epochs=100,
           #auto_lr_find=True


#            resume_from_checkpoint="model_checkpoints/yolov4epoch=82.ckpt",
        #    auto_lr_find=True,
          #  auto_scale_batch_size='binsearch',
        #    fast_dev_run=True
          )
t.fit(m)