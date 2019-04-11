from Pix2Pix import Pix2Pix

import tensorflow as tf
import numpy as np


if __name__ == "__main__":

    # Train the Augmentation Model using a cGAN

    # data_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate/imagesTr/'
    # target_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate/labelsTr/'
    #

    data_dir = 'D:/prostate_data/Task05_Prostate/imagesTr/'
    target_dir = 'D:/prostate_data/Task05_Prostate/labelsTr/'

    pretrained_model = 'ModelOutputs/UNetAugmentor_rev3/'
    ofolder = 'ModelOutputs/cGANUnetAugmentor_rev2/'

    tf.set_random_seed(1)
    np.random.seed(1)

    cgan = Pix2Pix(pretrained_folder=pretrained_model,
                   data_dir=data_dir,
                   target_dir=target_dir,
                   ofolder=ofolder,
                   batch_size=32)

    # cgan.train(epochs=20, sample_interval=1000)

    print('done')