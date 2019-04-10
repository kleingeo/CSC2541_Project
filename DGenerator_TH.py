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
import tensorflow as tf
import DGenUtils as Utils

from RandomAugmentation import RandomAugmentation


np.random.seed(1)
tf.set_random_seed(1)

class DGenerator_TH(keras.utils.Sequence):
    def __init__(self,
                 data_dir=r'D:\prostate_data\Task05_Prostate\imagesTr\\',
                 target_dir=r'D:\prostate_data\Task05_Prostate\labelsTr\\',
                 batch_size=20,
                 shuffle=False,
                 num_channels=1,
                 num_classes=1,
                 input_size=(128, 128),
                 regular=True,
                 augment_data=False):
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

        self.data_dir = data_dir
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.img_size = input_size
        self.reg = regular

        self.augment_data = augment_data

        # list all data to be used
        self.data_files = [''.join([self.data_dir, i]) for i in os.listdir(data_dir)]
        self.target_files = [''.join([self.target_dir, i]) for i in os.listdir(target_dir)]

        assert len(self.data_files) == len(self.target_files), 'Sample list and target list are different sizes.'

        self.training_data, self.target_data = Utils.PreprocessTrainingData(self.data_files,
                                                                            self.target_files,
                                                                            input_size)

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
        if isinstance(self.data_files, list) and isinstance(self.target_files, list):
            sample_list_temp = [self.training_data[k] for k in indexes]
            target_list_temp = [self.target_data[k] for k in indexes]

        # Generate data
        batch_x_data, batch_y_data = self.data_generation(sample_list_temp, target_list_temp)

        if self.reg == True:
            return batch_x_data, batch_y_data

        else:
            batch_x_data = (batch_x_data / 255).astype(np.float32)
            return batch_y_data, batch_x_data


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


        # batch_x_data = np.empty((self.batch_size,
        #                          self.img_size[0],
        #                          self.img_size[1],
        #                          self.num_channels
        #                          ))
        #
        # batch_y_data = np.empty((self.batch_size,
        #                          self.img_size[0],
        #                          self.img_size[1],
        #                          self.num_classes
        #                          ))

        # Generate data
        for i1 in range(len(sample_list)):
            batch_x_data[i1, 0, :, :] = sample_list[i1]
            # batch_y_data[i1, 0, :, :] = (target_list[i1] == 0).astype(np.uint8)
            batch_y_data[i1, 0, :, :] = target_list[i1]

            # x_data = sample_list[i1]
            #
            # y_data = target_list[i1]

            # if self.augment_data:
            #     rot_ang = (np.deg2rad(10))
            #     shear = (np.deg2rad(10))
            #     translate = True
            #     scale_factor = [0.8, 1.2]
            #     elastic = None
            #
            #     x_data, y_data = RandomAugmentation(
            #         img=x_data, seg_img=y_data,
            #         sample_size=self.img_size,
            #         rotation=rot_ang, scaling=scale_factor,
            #         translation=translate, shearing=shear, elastic=elastic)

            # batch_x_data[i1, :, :, 0] = x_data
            #
            # batch_y_data[i1, :, :, 0] = y_data

        return batch_x_data, batch_y_data

        # if self.reg ==True:
        #     return batch_x_data, batch_y_data
        # else:
        #     batch_x_data = (batch_x_data/255).astype(np.float32)
        #     return batch_y_data, batch_x_data

    def on_epoch_end(self):
        """
        Updates indices after each epoch
        """
        if self.shuffle is True:
            np.random.shuffle(self.idxs)


if __name__ == '__main__':
    a = DGenerator(regular=True)

    b = a.__getitem__(0)

    print('done')
