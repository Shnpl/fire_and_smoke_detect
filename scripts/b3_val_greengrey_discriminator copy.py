# train(val) the ir green grey discriminator -> modify the params manually, so there's no training process.
import sys
sys.path.append('.')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from models.ir_discriminator.green_grey_discriminator import GreenGreyDiscriminator_Litmodel

def main():
    hparams = {
        'batch_size':16,
    }
    trainer = pl.Trainer(devices=1,max_epochs=1,logger=False)
    
    model = GreenGreyDiscriminator_Litmodel(hparams)
    trainer.validate(model)
if __name__ == '__main__':
    main()