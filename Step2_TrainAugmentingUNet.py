from Trainer import Trainer
from Predictor import Predictor
import os
import keras as K

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    '''This script is used to train a basic UNet on prostate segmentation in 
    MRI. This will be used as a base line in our project. It is traained on 
    75% of the volumes, resampled to istropic voxel size of 2.0, resultingn 
    in 803 training examples. It is then tested on 25% of the volumes, 
    resampled the same, resulting in 278 testing examples. 
    '''


    tf.set_random_seed(1)
    np.random.seed(1)

    # Train the Augmentation Model

    data_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate' \
               '/imagesTr/'
    target_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate' \
                 '/labelsTr/'
    ofolder = 'ModelOutputs/UNetAugmentor_rev3/'


    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        CUDA_VISIBLE_DEVICES = ['None']


    a = Trainer(data_dir, target_dir, ofolder, samples_per_card=None,
                epochs=100, gpus_used=len(CUDA_VISIBLE_DEVICES),
                batch_size=16, training_direction=False)

    a.train_the_model(t_opt=K.optimizers.adam(lr=1e-4),
                      loss=K.losses.mae,
                      t_dropout=0)

    print('done')