from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import DGenerator as gen
from keras.models import model_from_json
import tensorflow as tf
from keras import backend as K
import TrainerFileNamesUtil as TrainerFileNamesUtil
K.set_image_dim_ordering('th')
plt.switch_backend('agg')

class Pix2Pix():
    def __init__(self,
                 data_dir=r'D:\prostate_data\Task05_Prostate\imagesTr\\',
                 target_dir=r'D:\prostate_data\Task05_Prostate\labelsTr\\',
                 batch_size = 50,
                 pretrained_folder=r'X:\2019-03-30-CSC2541Project'
                                   r'\UNetAugmentor\\',
                 ofolder=r'X:\2019-03-30-CSC2541Project\cGANUnetAugmentor\\'
                 ):

        self.pretrained_folder = pretrained_folder
        self.ofolder = ofolder
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.channels, self.img_rows, self.img_cols)
        self.classes = 2
        self.seg_shape = (self.classes, self.img_rows, self.img_cols)
        self.batch_size = batch_size

        # Configure data loader
        self.data_loader = gen.DGenerator(
            data_dir=data_dir,
            target_dir=target_dir,
            batch_size = self.batch_size,
            regular=False)


        # Calculate output shape of D (PatchGAN)
        self.disc_patch = (self.channels, 8, 8)

        # Number of filters in the first layer of G and D
        self.df = 64

        optimizer = Adam(0.0002, 0.5)
        # optimizer = Adam(1e-5, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.seg_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):

        json_file_name = [i for i in os.listdir(self.pretrained_folder) if i.endswith('json')][0]
        weights_file_name = [i for i in os.listdir(self.pretrained_folder) if i.startswith('model_best')][0]
        json_file = open(''.join([self.pretrained_folder, '/', json_file_name]))
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # Load the weights to the model
        model.load_weights(''.join([self.pretrained_folder, '/', weights_file_name]))

        model.name = 'generator'

        return model


    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.seg_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity, name='discriminator')

    def train(self, epochs, sample_interval=50):


        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((self.batch_size,) + self.disc_patch)
        fake = np.zeros((self.batch_size,) + self.disc_patch)

        batch_amt = self.data_loader.__len__()

        for epoch in range(epochs):
            for i in range(batch_amt):

                imgs_B, imgs_A = self.data_loader.__getitem__(i)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] "
                       "[G loss: %f] time: %s" % (epoch, epochs, i,
                                                  batch_amt,
                                                  d_loss[0], 100*d_loss[1],
                                                  g_loss[0], elapsed_time))

                # If at save interval => save generated image samples
                if i % sample_interval == 0:
                    self.sample_images(epoch, i)

        ## Save the model after training

        model_json = self.generator.to_json()

        with open(TrainerFileNamesUtil.create_model_json_filename(
                output_directory=self.ofolder,
                time_stamp=TrainerFileNamesUtil.get_time(),
                keyword='UNet_cGANAugmentor'),
                'w') as jason_file:
            jason_file.write(model_json)

        self.generator.save(self.ofolder + '/model_best_weights_.h5')

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % 'Prostate MRI ', exist_ok=True)
        r, c = 3, 3
        self.data_loader.batch_size = 3

        imgs_B, imgs_A = self.data_loader.__getitem__(0)

        fake_A = self.generator.predict(imgs_B)

        self.data_loader.batch_size = self.batch_size

        gen_imgs = np.concatenate([np.expand_dims(imgs_B[:, 1, :, :], axis=1),
                                   fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,0,:,:])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(self.ofolder +
                    "images%s%d_%d.png" % ('Prostate MRI', epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=1, batch_size=1, sample_interval=5)

    print('done')