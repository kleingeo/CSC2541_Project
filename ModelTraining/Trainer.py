'''
2018-09-28
(c) A. Martel Lab Co.
author: G.Kuling
This is my trainer code.
WARNING: This file is very hardcoded. Going to need to do some
        editing to generalize it.
'''

import sys
sys.path.append('..')

import numpy as np


import OtherModels.IUNet as IUNet
import OtherModels.UNet as UNet
import OtherModels.VGG as VGG
import OtherModels.ResNet as ResNet
import OtherModels.VNet as VNet
from keras.utils import multi_gpu_model
from OtherModels.Utils import dice_loss
import OtherModels.DataGenerator as generator
import OtherModels.MyCBK as MyCBK
import OtherModels.TrainerFileNamesUtil as TrainerFileNamesUtil
import pandas as pd
import sys


from log.logging_dict_configuration import logging_dict_config
import logging
from logging.config import dictConfig

class Histroies_Logger(Callback):

    def __init__(self, logger):

        super(Histroies_Logger, self).__init__()

        self.logger = logger


    def on_train_begin(self, logs={}):
        self.history = {'loss': [], 'val_loss': [], 'concurrency': [], 'val_concurrency': [], 'dice': [],
                        'val_dice': []}


    def on_epoch_end(self, epoch, logs={}):

        loss = logs['loss']
        val_loss = logs['val_loss']

        concurrancy = logs['concurrency']
        val_concurrancy = logs['val_concurrency']

        dice = logs['dice_coef']
        val_dice = logs['val_dice_coef']

        self.logger.info('Epoch {:d} - loss: {:.5f} - dice: {:.5f} - concurrency: {:.5f} - '
                         'val_loss: {:.5f} - val_dice: {:.5f} - val_concurrency: {:.5f}.'.format(epoch,
                                                                                                loss,
                                                                                                dice,
                                                                                                concurrancy,
                                                                                                val_loss,
                                                                                                val_dice,
                                                                                                val_concurrancy))


class ModelWeightSaver(Callback):

    def __init__(self, single_gpu_model, weights_filename, period, total_epoch_size):

        super(ModelWeightSaver, self).__init__()

        self.single_gpu_model = single_gpu_model
        self.weights_filename = weights_filename
        self.period = period
        self.total_epoch_size = total_epoch_size

    def on_epoch_end(self, epoch, logs=None):

        if self.period is not None:

            if ((epoch + 1) % self.period == 0) and ((epoch + 1) != self.total_epoch_size):

                name = self.weights_filename + '_' + str(epoch + 1) + '_weights.h5'

                self.single_gpu_model.save(name)
        #
        # if (epoch + 1) == self.total_epoch_size:
        #
        #     name = self.weights_filename + '_' + str(epoch + 1) + '_weights.h5'
        #
        #     self.single_gpu_model.save(name)



class Trainer():
    def __init__(self,
                 img_filelist,
                 label_filelist,
                 file_path,
                 batch_size=5,
                 sample_size=(256, 256, 1),
                 shuffle=True,
                 augment=False,
                 ofolder=None,
                 samples_per_card=None,
                 epochs=50,
                 gpus_used=1,
                 model_type='IUNet'):
        self.img_filelist = img_filelist
        self.label_filelist = label_filelist
        self.file_path = file_path
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.augment = augment
        self.ofolder = ofolder
        self.samples_per_card = samples_per_card
        self.epochs = epochs
        self.gpus_used = gpus_used
        self.model_type = model_type


    def train_the_model(self):
        """
        Main driver of the trainer
        :param t_opt: Optimizer option
        :param t_dilRate: dilation rate
        :param t_depth: depth
        :param t_dropOut: drop out amount
        :return:
        """
        ### Main code
        # start up the data generator
        gen = generator.data_generator(self.img_filelist,
                                       self.label_filelist,
                                       self.file_path,
                                       self.batch_size,
                                       (self.sample_size[0],
                                        self.sample_size[1]),
                                       self.shuffle,
                                       self.augment)

        # load the model
        if self.model_type is 'IUNet':
            model = IUNet.get_iunet(img_x = self.sample_size[0],
                                    img_y = self.sample_size[1],
                                    dilation_rate = 1,
                                    depth = 5,
                                    base_filter = 16,
                                    batch_normalization = False,
                                    pool_1d_size = 2,
                                    deconvolution = False,
                                    dropout = 0.0,
                                    num_classes=1,
                                    num_seq=self.sample_size[-1])
        if self.model_type is 'UNet':
            model = UNet.get_unet(img_x = self.sample_size[0],
                                  img_y = self.sample_size[1],
                                  dilation_rate = 1,
                                  depth = 5,
                                  base_filter = 16,
                                  batch_normalization = False,
                                  pool_1d_size = 2,
                                  deconvolution = False,
                                  dropout = 0.0,
                                  num_classes=1,
                                  num_seq=self.sample_size[-1])
        if self.model_type is 'VGG':
            model = VGG.get_vgg(img_x = self.sample_size[0],
                                img_y = self.sample_size[1],
                                dropout = 0.0,
                                num_classes = 1,
                                num_seq = self.sample_size[-1])
        if self.model_type is 'ResNet':
            model = ResNet.get_resnet(img_x = self.sample_size[0],
                                      img_y = self.sample_size[1],
                                      f=16,
                                      bn_axis=3,
                                      num_classes=1,
                                      num_seq = self.sample_size[-1])
        if self.model_type is 'VNet':
            model = VNet.get_vnet(img_x = self.sample_size[0],
                                  img_y = self.sample_size[1],
                                  dilation_rate = 1,
                                  depth = 5,
                                  batch_normalization = False,
                                  deconvolution = False,
                                  dropout = 0.0,
                                  num_classes=1,
                                  num_seq=self.sample_size[-1])


        # setup a multi GPU trainer
        if self.gpus_used>1:
            gmodel = multi_gpu_model(model, gpus=self.gpus_used)
            gmodel.compile(optimizer='ADAM',
                           loss=dice_loss,
                           metrics=[dice_loss])
            cbk = MyCBK.MyCBK(model, self.ofolder)

            # begin training
            gmodel.fit_generator(gen,
                                 epochs=self.epochs,
                                 verbose=1,
                                 steps_per_epoch=100,
                                 workers=20,
                                 use_multiprocessing=True,
                                 callbacks=[cbk])
        else:
            gmodel = model
            gmodel.compile(optimizer='ADAM',
                           loss=dice_loss,
                           metrics=[dice_loss])
            cbk = MyCBK.MyCBK(model, self.ofolder)

            # begin training
            gmodel.fit_generator(gen,
                                 epochs=self.epochs,
                                 verbose=1,
                                 steps_per_epoch=100,
                                 callbacks=[cbk])

        # model check points, save the best weights

        model_json = model.to_json()
        with open(TrainerFileNamesUtil.create_model_json_filename(
                output_directory=self.ofolder,
                time_stamp=TrainerFileNamesUtil.get_time(),
                keyword=self.model_type), 'w') as jason_file:
            jason_file.write(model_json)

        print('Training is finished!')


if __name__ == "__main__":
    ### controlable parameters
    # For oberon
    # batch_folder = '/jaylabs/amartel_data2/Grey/2018-06-11-FGTSeg-Data' \
    #                '/Batches/Training/data/'
    # target_folder = '/jaylabs/amartel_data2/Grey/2018-06-11-FGTSeg-Data' \
    #                '/Batches/Training/target/'
    # ofolder = '/jaylabs/amartel_data2/Grey/2018-06-11-FGTSeg-Data/Models' \
    #           '/RedoFolder/'

    # For Titania
    # batch_folder = '/labs/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
    #                '/Batches/2D/Training/data/'
    # target_folder = '/labs/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
    #                 '/Batches/2D/Training/target/'
    # ofolder = '/labs/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data/Models' \
    #           '/SingleClass2D/'

    file_path = '../prostate_data'

    df = pd.read_pickle('../build_dataframe/dataframe_slice.pickle')

    img_filelist = df['image_filename'].loc[df['train_val_test'] == 'train'].values

    label_filelist = df['label_filename'].loc[df['train_val_test'] == 'train'].values

    ofolder = '../OtherModels/model_output'

    a = Trainer(img_filelist,
                label_filelist,
                file_path,
                batch_size=5,
                sample_size=(256, 256, 1),
                shuffle=True,
                augment=False,
                ofolder=ofolder,
                samples_per_card=5,
                epochs=1,
                gpus_used=1,
                model_type='IUNet')

    a.train_the_model()


    print('done')