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
from keras.callbacks import ModelCheckpoint, Callback
import os
import pandas as pd
import keras
import tensorflow as tf

from ModelTraining.ModelSelectUtil import ModelSelectUtil
from ModelTraining.ModelParamUtil import ModelParamUtil
import ModelTraining.TrainerFileNamesUtil as TrainerFileNamesUtil

from keras.utils import multi_gpu_model
from Dataset.DataGenerator import DataGenerator


from OtherModels.Utils import dice_loss, dice


import ModelTraining.GridSearchUtil as grid_util
import ModelTraining.GridSearch_Consts as GS_Util






from log.logging_dict_configuration import logging_dict_config
import logging
from logging.config import dictConfig

class Histroies_Logger(Callback):

    def __init__(self, logger):

        super(Histroies_Logger, self).__init__()

        self.logger = logger


    def on_train_begin(self, logs={}):
        self.history = {'loss': [], 'val_loss': [], 'dice': [], 'val_dice': []}


    def on_epoch_end(self, epoch, logs={}):

        loss = logs['loss']
        val_loss = logs['val_loss']

        dice = logs['dice']
        val_dice = logs['val_dice']

        self.logger.info('Epoch {:d} - loss: {:.5f} - dice: {:.5f} - '
                         'val_loss: {:.5f} - val_dice: {:.5f}.'.format(epoch,
                                                                       loss,
                                                                       dice,
                                                                       val_loss,
                                                                       val_dice))


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




class Trainer():

    def __init__(self,
                 output_directory,
                 t2_img_filelist_train,
                 seg_filelist_train,
                 seg_slice_train,
                 t2_img_filelist_val,
                 seg_filelist_val,
                 seg_slice_val,
                 t2_file_path,
                 seg_file_path,
                 t1_img_filelist_train=None,
                 flair_img_filelist_train=None,
                 t1_img_filelist_val=None,
                 flair_img_filelist_val=None,
                 t1_file_path=None,
                 flair_file_path=None,
                 batch_size=5,
                 sample_size=(256, 256),
                 shuffle=True,
                 epochs=50,
                 model_type='IUNet',
                 multi_gpu=False,
                 training_log_name=None,
                 train_with_fake=False,
                 train_fraction=1.0,
                 trainer_grid_search=None,
                 relative_save_weight_peroid=None):


        self.top_output_directory = output_directory

        self.t2_img_filelist_train = t2_img_filelist_train
        self.t2_img_filelist_val = t2_img_filelist_val
        self.t2_file_path = t2_file_path

        self.seg_filelist_train = seg_filelist_train
        self.seg_filelist_val = seg_filelist_val
        self.seg_slice_train = seg_slice_train
        self.seg_slice_val = seg_slice_val
        self.seg_file_path = seg_file_path


        self.t1_img_filelist_train = t1_img_filelist_train
        self.t1_img_filelist_val = t1_img_filelist_val
        self.t1_file_path = t1_file_path

        self.flair_filelist_train = flair_img_filelist_train
        self.flair_img_filelist_val = flair_img_filelist_val
        self.flair_file_path = flair_file_path


        self.sample_size = sample_size

        self.shuffle = shuffle

        self.epochs = epochs
        self.batch_size = batch_size

        self.model_type = model_type
        self.multi_gpu = multi_gpu

        self.num_channels = 1

        self.train_with_fake = train_with_fake

        self.train_fraction = train_fraction

        self.relative_save_weight_peroid = relative_save_weight_peroid

        if t1_img_filelist_train is not None:
            self.num_channels = self.num_channels + 1

        if flair_img_filelist_train is not None:
            self.num_channels = self.num_channels + 1


        self.time_stamp = TrainerFileNamesUtil.get_time()

        self.time_stamp_dir = os.path.join(self.top_output_directory, self.time_stamp)

        if not os.path.exists(self.time_stamp_dir):
            os.makedirs(self.time_stamp_dir)


        self.training_log_name = training_log_name


        if self.training_log_name is None:

            self.training_log_name = self.top_output_directory + '/' + self.time_stamp + '/train.log'

        self.logging_dict = logging_dict_config(self.training_log_name)

        dictConfig(self.logging_dict)

        self.logger = logging.getLogger(__name__)

        self.logger_format = logging.Formatter(fmt=self.logging_dict['formatters']['f']['format'],
                                               datefmt=self.logging_dict['formatters']['f']['datefmt'])



        if trainer_grid_search is None:


            params_dictionary = dict(model_typpe=self.model_type,
                                     epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     train_with_fake=self.train_with_fake,
                                     train_fraction=self.train_fraction)

            grid_utility = grid_util.GridSearchUtil(params_dictionary)
            self.grid_search_params = grid_utility.get_params_dataframe()

        else:

            params_dictionary = trainer_grid_search

            grid_utility = grid_util.GridSearchUtil(params_dictionary)
            self.grid_search_params = grid_utility.get_params_dataframe()
            grid_utility.save(output_location=self.top_output_directory + '/' + self.time_stamp)



    def build_model_param_directory(self, model_param):

        self.keyword = TrainerFileNamesUtil.build_filename_keyword(model_param)

        self.output_directory = TrainerFileNamesUtil.create_output_directory(
            self.top_output_directory,
            os.path.join(self.time_stamp, self.keyword))



    def build_filenames(self):


        self.model_output_directory = self.output_directory


        self.model_json_filename = TrainerFileNamesUtil.create_model_json_filename(
                self.model_output_directory,
                self.keyword)

        self.model_weights_filename = TrainerFileNamesUtil.create_model_weights_filename(
                self.model_output_directory,
                self.keyword)


        self.training_history_filename = TrainerFileNamesUtil.create_training_history_filename(
                self.model_output_directory,
                self.keyword)


        self.logger.info('model json file is ' + str(self.model_json_filename))
        self.logger.info(
            'model weights file is ' + str(self.model_weights_filename))
        self.logger.info(
            'training history file is ' + str(self.training_history_filename))


    def save_training_history(self):
        """
        this method saves the training history to a file
        """

        try:
            self.history_df.to_csv(self.training_history_filename, index=False)
            self.logger.info(
                'history was saved successfully to ' + self.training_history_filename)
        except:
            self.logger.error(
                'history file could not be saved to ' + self.training_history_filename)


    def save_model_to_json(self):
        """
        this method saves the model's json file, required for loading the
        model later, properly
        """
        model_json = self.model.to_json()
        with open(self.model_json_filename, 'w') as jason_file:
            jason_file.write(model_json)
            self.logger.info(
                'model saved successfully to ' + self.model_json_filename)


    def train(self):
        """
        Main driver of the trainer

        """


        self.logger.info('Beginning to train model.')

        for index, _ in self.grid_search_params.iterrows():

            train_params = self.grid_search_params.loc[index]

            batch_size = train_params[GS_Util.BATCH_SIZE()]
            epoch_size = train_params[GS_Util.EPOCHS()]


            # batch_size = train_params['batch_size']
            # epoch_size = train_params['epochs']

            self.build_model_param_directory(train_params)

            if index > 0:
                self.logger.removeHandler(handler)

            handler = logging.FileHandler(self.output_directory + '/' + 'training.log')

            handler.setFormatter(self.logger_format)

            self.logger.addHandler(handler)

            self.logger.info('model parameters are: ')
            self.logger.info(train_params)

            self.build_filenames()


            self.logger.info(
                ('model parameters number ' + str(index) + ' of ' +
                 str(self.grid_search_params.index)))

            # load the model


            self.model_fn = ModelSelectUtil(train_params[GS_Util.MODEL_TYPE()])

            model_params = ModelParamUtil(train_params[GS_Util.MODEL_TYPE()])


            model_params['img_shape'] = self.sample_size

            train_size = int(len(self.t2_img_filelist_train) * train_params[GS_Util.TRAIN_FRAC()])

            if train_params[GS_Util.WITH_FAKE()]:

                model_params['num_channels'] = 3

                params_train_generator = {'sample_size': self.sample_size,
                                          'batch_size': batch_size,
                                          'n_channels': 3,
                                          'shuffle': True,
                                          'augment_data': train_params[GS_Util.AUGMENT_TRAINING()]}

                params_val_generator = {'sample_size': self.sample_size,
                                        'batch_size': batch_size,
                                        'n_channels': 3,
                                        'shuffle': False,
                                        'augment_data': False}


                training_generator = DataGenerator(
                    t2_sample=self.t2_img_filelist_train[:train_size],
                    seg_sample=self.seg_filelist_train[:train_size],
                    t2_sample_main_path=self.t2_file_path,
                    seg_sample_main_paths=self.seg_file_path,
                    seg_slice_list=self.seg_slice_train[:train_size],
                    t1_sample=self.t1_img_filelist_train[:train_size],
                    flair_sample=self.flair_filelist_train[:train_size],
                    t1_sample_main_path=self.t1_file_path,
                    flair_sample_main_path=self.flair_file_path,
                    **params_train_generator)

                validation_generator = DataGenerator(
                    t2_sample=self.t2_img_filelist_val,
                    seg_sample=self.seg_filelist_val,
                    t2_sample_main_path=self.t2_file_path,
                    seg_sample_main_paths=self.seg_file_path,
                    seg_slice_list=self.seg_slice_val,
                    t1_sample=self.t1_img_filelist_val,
                    flair_sample=self.flair_img_filelist_val,
                    t1_sample_main_path=self.t1_file_path,
                    flair_sample_main_path=self.flair_file_path,
                    **params_val_generator)

            else:

                model_params['num_channels'] = 1

                params_train_generator = {'sample_size': self.sample_size,
                                          'batch_size': batch_size,
                                          'n_channels': 1,
                                          'shuffle': True,
                                          'augment_data': train_params[GS_Util.AUGMENT_TRAINING()]}

                params_val_generator = {'sample_size': self.sample_size,
                                        'batch_size': batch_size,
                                        'n_channels': 1,
                                        'shuffle': False,
                                        'augment_data': False}

                training_generator = DataGenerator(
                    t2_sample=self.t2_img_filelist_train[:train_size],
                    seg_sample=self.seg_filelist_train[:train_size],
                    t2_sample_main_path=self.t2_file_path,
                    seg_sample_main_paths=self.seg_file_path,
                    seg_slice_list=self.seg_slice_train[:train_size],
                    **params_train_generator)

                validation_generator = DataGenerator(
                    t2_sample=self.t2_img_filelist_val,
                    seg_sample=self.seg_filelist_val,
                    t2_sample_main_path=self.t2_file_path,
                    seg_sample_main_paths=self.seg_file_path,
                    seg_slice_list=self.seg_slice_val,
                    **params_val_generator)


            self.model = self.model_fn(**model_params)

            self.logger.info('model has been built successfully')

            self.optimizer = keras.optimizers.Adam(lr=1e-5)

            if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
                CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
            else:
                CUDA_VISIBLE_DEVICES = ['None']

            if self.multi_gpu == True and len(CUDA_VISIBLE_DEVICES) > 1:

                self.model_parallel = multi_gpu_model(self.model, gpus=len(CUDA_VISIBLE_DEVICES))
                self.model_parallel.compile(optimizer=self.optimizer, loss=dice_loss, metrics=[dice])

            else:
                self.model.compile(optimizer=self.optimizer, loss=dice_loss, metrics=[dice])

                self.model_parallel = self.model


            if self.relative_save_weight_peroid is not None:
                period = int(epoch_size / self.relative_save_weight_peroid)
            else:
                period = None


            model_weight_saver_callback = ModelWeightSaver(self.model, self.model_weights_filename, period=period,
                                                           total_epoch_size=epoch_size)

            model_logger = Histroies_Logger(self.logger)

            self.logger.info('training started')

            self.save_model_to_json()


            history = self.model_parallel.fit_generator(
                generator=training_generator,
                steps_per_epoch=int(len(self.t2_img_filelist_train) / batch_size),
                epochs=epoch_size,
                callbacks=[model_weight_saver_callback, model_logger],
                validation_data=validation_generator,
                validation_steps=int(len(self.t2_img_filelist_val) / batch_size),
                verbose=1,
                shuffle=True,
                use_multiprocessing=True,
                workers=30,
                max_queue_size=30)

            self.model.save(self.model_weights_filename + '_' + str(epoch_size) + '_weights.h5')

            self.history_df = pd.DataFrame(history.history)

            self.logger.info(
                'training completed for this set of parameters.')

            self.save_training_history()

            self.logger.info(
                self.model_json_filename + ' is successfully saved.')
            self.logger.info(
                self.training_history_filename + ' is successfully saved.')