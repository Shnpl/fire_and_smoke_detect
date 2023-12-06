# test the ir discriminator
import sys
sys.path.append('.')
import yaml

import pytorch_lightning as pl

from models.ir_discriminator.ir_discriminator import IRDiscriminator_Litmodel

def main():
    path = 'lightning_logs/ir_discriminator/version_0'
    epoch = 2
    with open(f'{path}/hparams.yaml') as f:
        hparams = yaml.load(f,Loader=yaml.FullLoader)
    model = IRDiscriminator_Litmodel.load_from_checkpoint(f'{path}/checkpoints/epoch={epoch}.ckpt')
    trainer = pl.Trainer(devices=1,logger=False)
    trainer.test(model)    
   
if __name__ == '__main__':
    main()