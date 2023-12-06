# train the ir discriminator
import sys
sys.path.append('.')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from models.ir_discriminator.ir_discriminator import IRDiscriminator_Litmodel

def main():
    hparams = {
        'batch_size':16,
        'loss':'cross_entropy',
        'lr':1e-3,
        'max_epochs':3
    }
    logger = TensorBoardLogger(save_dir='lightning_logs',name='ir_discriminator')
    trainer = pl.Trainer(devices=1,max_epochs=hparams['max_epochs'],logger=logger)
    model = IRDiscriminator_Litmodel(hparams)
    trainer.fit(model)
if __name__ == '__main__':
    main()