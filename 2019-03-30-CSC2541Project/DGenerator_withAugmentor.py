"""
2018-09-28
(c) A.Martel Lab Co.
author: G.Kuling
Description: This is a data generator that will be used to process batches
that are made to train a network to segment fat tissue and fibroglandular
tissue in multi weighted breast MRI.
WARNING: This file is very hardcoded. Going to need to do some
        editing to generalize it.
"""

import keras
import numpy as np
import os
import DGenUtils as Utils
from keras.models import model_from_json
from TrainingUtils import My_new_loss
from keras.losses import mse
np.random.seed(1)


class DGenerator_withAugmentor(keras.utils.Sequence):
    def __init__(self,
                 data_dir=r'D:\prostate_data\Task05_Prostate\imagesTr',
                 target_dir=r'D:\prostate_data\Task05_Prostate\labelsTr',
                 aug_folder = r'X:\2019-03-30-CSC2541Project\UNetAugmentor',
                 batch_size = 20,
                 shuffle = True,
                 num_channels = 1,
                 num_classes = 2,
                 input_size = (128, 128),
                 regular = True):
        """
        Initialization for data generator
        :param data_dir: (dir of numpy arrays) Directory training data is saved.
            Individual samples in each npy array.
        :param target_dir: (dir of numpy arrays) Directory training targets are
        saved. Individual samples in each npy array.
        :param batch_size: (int) Size of batches to be produced each time
            __getitem__ is called
        :param mode: (str) mode of input type. Option=['2Ch', 'WOFS', 'FS']
        :param img_size: (list of ints) the dimensions of the input
        :param shuffle: (bool) whether to shuffle training examples between
            each epoch
        :param num_classes: (int) number of classes that are labeled in the
            target masks
        """

        json_file_name = \
            [i for i in os.listdir(aug_folder) if i.endswith('json')][0]
        weights_file_name = \
            [i for i in os.listdir(aug_folder) if i.startswith(
                'model_best')][0]
        json_file = open(''.join([aug_folder, '/', json_file_name]))
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # Load the weights to the model
        model.load_weights(''.join([aug_folder, '/', weights_file_name]))
        model.compile(loss=mse, metrics=[mse],
                           optimizer='ADAM')
        print('Model is ready to synthesize data.')

        self.data_dir = data_dir
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.img_size = input_size
        self.reg = regular


        # list all data to be used
        self.data_files = [''.join([self.data_dir,i]) for i in
                           os.listdir(data_dir)]
        self.target_files = [''.join([self.target_dir, i]) for i in
                           os.listdir(target_dir)]
        assert len(self.data_files) == len(self.target_files), \
            'Sample list and target list are different sizes.'

        self.training_data, self.target_data = Utils.PreprocessTrainingData(
            self.data_files, self.target_files, input_size)

        # Build atlas for the augmentor UNet
        atlas = np.zeros(input_size)

        for i in range(len(self.target_data)):
            atlas = atlas + self.target_data[i]

        atlas = np.divide(atlas, len(self.target_data))
        n_sam = len(self.training_data)
        thrs = np.random.random((n_sam))
        for i in range(n_sam):
            map = np.zeros((1,2,128,128))
            map[0,1,:,:] = (atlas>thrs[i]).astype('uint8')
            map[0,0,:,:] = np.ones((128, 128)) - (atlas>thrs[i]).astype('uint8')
            img = model.predict(map)

            self.training_data.append(np.around(img[0,0,:,:]*255))
            self.target_data.append(map[0,1,:,:])

        self.idxs = np.arange(len(self.training_data))
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.ceil(len(self.training_data) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.idxs[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Find list of samples and corresponding targets
        if isinstance(self.data_files, list) and \
                isinstance(self.target_files, list):
            sample_list_temp = [self.training_data[k] for k in indexes]
            target_list_temp = [self.target_data[k] for k in indexes]

        # Generate data
        batch_x_data, batch_y_data = self.data_generation(sample_list_temp,
                                                            target_list_temp)

        ### If you need to change data types, do this here
        batch_x_data = batch_x_data.astype(np.float32)
        batch_y_data = batch_y_data.astype(np.uint8)

        return batch_x_data, batch_y_data

    def data_generation(self, sample_list, target_list):
        """
        Generates the data into batches to be passed to the trainer.
        :param sample_list: (list) list of the training examples to be imported
        :param target_list: (list) corresponding targets for the training list
        :return: a bathc of training images and a batch of target segmentations
        """
        # Initialization
        batch_x_data = np.empty((self.batch_size,
                                 self.num_channels,
                                 self.img_size[0],
                                 self.img_size[1]
                                 ))

        batch_y_data = np.empty((self.batch_size,
                                 self.num_classes,
                                 self.img_size[0],
                                 self.img_size[1]
                                 ))

        # Generate data
        for i1 in range(len(sample_list)):
            batch_x_data[i1,0,:,:] = sample_list[i1]
            batch_y_data[i1,0,:,:] = (target_list[i1]==0).astype(np.uint8)
            batch_y_data[i1, 1, :, :] = target_list[i1]
        if self.reg ==True:
            return batch_x_data, batch_y_data
        else:
            return batch_y_data, batch_x_data

    def on_epoch_end(self):
        """
        Updates indices after each epoch
        """
        if self.shuffle is True:
            np.random.shuffle(self.idxs)


if __name__ == '__main__':
    a = DGenerator_withAugmentor(regular=True)

    b = a.__getitem__(0)

    print('done')
