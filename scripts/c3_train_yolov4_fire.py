import sys
sys.path.append('.')
from models.Yet_Another_YOLOv4_Pytorch.pl_model import YOLOv4PL
from lightning.pytorch.utilities.parsing import AttributeDict
import pytorch_lightning as pl
from argparse import Namespace
import torch
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
# # pretrain with no tricks
# hparams = {
#     "n_classes" : 2,
#     "weights_path": "lightning_logs/yolov4/yolov4_pretrain.pth",
#     "pretrained" : True,
#     "train_ds" : "datasets/fire_and_smoke_detect/Fire/train/train.txt",
#     "valid_ds" : "datasets/fire_and_smoke_detect/Fire/val/val.txt",
#     "img_extensions" : [".JPG", ".jpg"],
#     "bs" : 4,
#     "momentum": 0.9,
#     "wd": 0.001,
#     "lr": 1e-4,
#     "epochs" : 100,
#     "pct_start" : 10/100,
    
#     "optimizer" : "Ranger",
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
# pretrain with no tricks,lr=1e-5
# hparams = {
#     "n_classes" : 2,
#     "weights_path": "lightning_logs/yolov4/yolov4_pretrain.pth",
#     "pretrained" : True,
#     "train_ds" : "datasets/fire_and_smoke_detect/Fire/train/train.txt",
#     "valid_ds" : "datasets/fire_and_smoke_detect/Fire/val/val.txt",
#     "img_extensions" : [".JPG", ".jpg"],
#     "bs" : 4,
#     "momentum": 0.9,
#     "wd": 0.001,
#     "lr": 1e-5,
#     "epochs" : 100,
#     "pct_start" : 10/100,
    
#     "optimizer" : "Ranger",
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
# # pretrain with no tricks,lr=1e-5,and Adam optimizer
# hparams = {
#     "n_classes" : 2,
#     "weights_path": "lightning_logs/yolov4/yolov4_pretrain.pth",
#     "pretrained" : True,
#     "train_ds" : "datasets/fire_and_smoke_detect/Fire/train/train.txt",
#     "valid_ds" : "datasets/fire_and_smoke_detect/Fire/val/val.txt",
#     "img_extensions" : [".JPG", ".jpg"],
#     "bs" : 4,
#     "momentum": 0.9,
#     "wd": 0.001,
#     "lr": 1e-5,
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
# # pretrained, no tricks, lr=1e-4,Adam
hparams = {
    "n_classes" : 2,
    "weights_path": "lightning_logs/yolov4/yolov4_pretrain.pth",
    "pretrained" : True,
    "train_ds" : "datasets/fire_and_smoke_detect/Fire/train/train.txt",
    "valid_ds" : "datasets/fire_and_smoke_detect/Fire/val/val.txt",
    "img_extensions" : [".JPG", ".jpg"],
    "bs" : 4,
    "momentum": 0.9,
    "wd": 0.001,
    "lr": 1e-4,
    "epochs" : 100,
    "pct_start" : 10/100,
    
    "optimizer" : "Adam",
    "flat_epochs" : 50,
    "cosine_epochs" : 25,
    "scheduler" : "Cosine Delayed", 
    
    "SAT" : True,
    "epsilon" : 0.1,
    "SAM" : False,
    "ECA" : False,
    "WS" : False,
    "Dropblock" : False,
    "iou_aware" : False,
    "coord" : False,
    "hard_mish" : False,
    "asff" : False,
    "repulsion_loss" : False,
    "acff" : True,
    "bcn" : False,
    "mbn" : False,
    "anchors":[[[20, 35],[45, 68],[90, 97]],
               [[85, 198],[206, 130],[166,238]],
               [[223, 426],[372, 260],[488, 475]]],
    "train_part":['backbone','neck','head'],
}

# pretrained, few tricks, lr=1e-4,Adam
# hparams = {
#     "n_classes" :1,
#     "weights_path": "lightning_logs/yolov4/yolov4_pretrain.pth",
#     "pretrained" : True,
#     "train_ds" : "datasets/fire_and_smoke_detect/Fire/train/train.txt",
#     "valid_ds" : "datasets/fire_and_smoke_detect/Fire/val/val.txt",
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
    
#     "SAT" : "fgsm",
#     "epsilon" : 0.1,
#     "SAM" : True,
#     "ECA" : False,
#     "WS" : True,
#     "Dropblock" : True,
#     "iou_aware" : True,
#     "coord" : True,
#     "hard_mish" : False,# ->Use Mish
#     "asff" : False,
#     "repulsion_loss" : False,
#     "acff" : False,
#     "bcn" : False,
#     "mbn" : False,
# }

# pretrain with all tricks
# hparams = {
#     "n_classes" : 2,
#     "weights_path": "lightning_logs/yolov4/yolov4_pretrain.pth",
#     "pretrained" : True,
#     "train_ds" : "datasets/fire_and_smoke_detect/Fire/train/train.txt",
#     "valid_ds" : "datasets/fire_and_smoke_detect/Fire/val/val.txt",
#     "img_extensions" : [".JPG", ".jpg"],
#     "bs" : 2,
#     "momentum": 0.9,
#     "wd": 0.001,
#     "lr": 1e-4,
#     "epochs" : 100,
#     "pct_start" : 10/100,
    
#     "optimizer" : "Ranger",
#     "flat_epochs" : 50,
#     "cosine_epochs" : 25,
#     "scheduler" : "Cosine Delayed", 
    
#     "SAT" : True,
#     "epsilon" : 0.1,
#     "SAM" : True,
#     "ECA" : True,
#     "WS" : True,
#     "Dropblock" : True,
#     "iou_aware" : True,
#     "coord" : True,
#     "hard_mish" : True,
#     "asff" : True,
#     "repulsion_loss" : False,
#     "acff" : False,
#     "bcn" : True,
#     "mbn" : True,
# }

hparams = AttributeDict(hparams)
m = YOLOv4PL(hparams)
wandb_logger = WandbLogger(save_dir='lightning_logs/', name = "yolov4_fire_sat",project='Fire_and_Smoke_Detection')
checkpoint_callback = ModelCheckpoint(
    every_n_epochs=1,
    save_top_k=-1,
    verbose=True,
)
torch.autograd.set_detect_anomaly(True)
t = pl.Trainer(logger = wandb_logger,
           devices=[1],
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