import keras
import keras.backend as K
import keras.models as KM
import keras.layers as KL
from keras.layers import advanced_activations
import tensorflow as tf


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime

from MutliChannelGAN.DataGenerator import data_generator

class MuiltiChannelGAN():

    def __init__(self):

        tf.random.set_random_seed(1)
        np.random.seed(10)

        self.batch_size = 30
        self.sample_size = (256, 256)
        self.number_channels = 1

        self.img_shape = self.sample_size + (self.number_channels,)

        self.latent_dim = 100

        patch = int(self.sample_size[0] / 2**4)
        self.disc_patch = (patch, patch, 1)


        self.gf = 64
        self.df = 64

        optimizer = keras.optimizers.Adam(0.0002, 0.5)

        # optimizer = keras.optimizers.Adam(1e-4)

        # Build and compile the discriminator

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])




        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = KL.Input(shape=(self.latent_dim,))




        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False


        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = KM.Model(z, validity)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=optimizer)



    def build_generator(self):

        momentum = 0.99

        noise = KL.Input(shape=(self.latent_dim,))

        x = KL.Dense(512 * 16 * 16)(noise)
        x = KL.Reshape((16, 16, 512))(x)
        x = KL.BatchNormalization()(x)

        x = KL.Conv2DTranspose(256, 5, strides=2, padding='same')(x)
        x = KL.BatchNormalization(momentum=momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2DTranspose(128, 5, strides=2, padding='same')(x)
        x = KL.BatchNormalization(momentum=momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2DTranspose(64, 5, strides=2, padding='same')(x)
        x = KL.BatchNormalization(momentum=momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2DTranspose(32, 5, strides=2, padding='same')(x)
        x = KL.BatchNormalization(momentum=momentum)(x)
        x = KL.Activation('relu')(x)

        x = KL.Conv2D(self.number_channels, 5, padding='same')(x)
        x = KL.Activation('tanh')(x)

        return KM.Model(noise, x)


    def build_discriminator(self):


        img = KL.Input(shape=self.img_shape)

        momentum = 0.99
        dropout = 0.3

        # x = KL.Conv2D(64, 3, strides=2, padding='same')(img)
        # x = KL.LeakyReLU(alpha=0.2)(x)
        #
        # x = KL.Conv2D(128, 3, strides=2, padding='same')(x)
        # x = KL.LeakyReLU(alpha=0.2)(x)
        #
        # x = KL.Conv2D(256, 3, strides=2, padding='same')(x)
        # x = KL.LeakyReLU(alpha=0.2)(x)
        #
        # x = KL.Conv2D(512, 3, strides=2, padding='same')(x)
        # x = KL.LeakyReLU(alpha=0.2)(x)
        #
        #
        # x = KL.Conv2D(1, 3, padding='same')(x)



        x = KL.Conv2D(32, 5, strides=2, padding='same')(img)
        x = KL.LeakyReLU(alpha=0.2)(x)
        # x = KL.Dropout(dropout)(x)

        x = KL.Conv2D(64, 5, strides=2, padding='same')(x)
        x = KL.BatchNormalization(momentum=momentum)(x)
        x = KL.LeakyReLU(alpha=0.2)(x)
        # x = KL.Dropout(dropout)(x)

        x = KL.Conv2D(128, 5, strides=2, padding='same')(x)
        x = KL.BatchNormalization(momentum=momentum)(x)
        x = KL.LeakyReLU(alpha=0.2)(x)
        # x = KL.Dropout(dropout)(x)


        x = KL.Conv2D(256, 5, strides=2, padding='same')(x)
        x = KL.BatchNormalization(momentum=momentum)(x)
        x = KL.LeakyReLU(alpha=0.2)(x)
        # x = KL.Dropout(dropout)(x)

        x = KL.Conv2D(512, 5, strides=2, padding='same')(x)
        x = KL.BatchNormalization(momentum=momentum)(x)
        x = KL.LeakyReLU(alpha=0.2)(x)
        # x = KL.Dropout(dropout)(x)

        x = KL.Flatten()(x)

        x = KL.Dense(1, activation='sigmoid')(x)

        return KM.Model(img, x)


    def train(self):

        start_time = datetime.datetime.now()

        epochs = 10

        sample_interval = 5

        file_path = '../prostate_data'
        df = pd.read_pickle('../build_dataframe/dataframe_slice.pickle')
        img_filelist = df['image_filename'].loc[df['train_val_test'] == 'train'].values

        label_filelist = df['label_filename'].loc[df['train_val_test'] == 'train'].values



        batch_step_size = int(len(img_filelist) / self.batch_size)

        train_gen = data_generator(img_filelist[:batch_step_size * self.batch_size],
                                   label_filelist[:batch_step_size * self.batch_size],
                                   file_path, batch_size=self.batch_size, sample_size=self.sample_size,
                                   shuffle=True, augment=False)


        # Adversarial loss ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))



        for epoch in range(epochs):
            for batch_i in range(batch_step_size):

                # Select a random batch of images
                imgs = next(train_gen)


                # plt.imshow(imgs[0, :, :, 0])
                # plt.show()
                #
                # plt.imshow(imgs[40, :, :, 0])
                # plt.show()
                #
                # plt.imshow(imgs[25, :, :, 0])
                # plt.show()
                #
                # plt.imshow(imgs[30, :, :, 0])
                # plt.show()
                #
                # plt.imshow(imgs[10, :, :, 0])
                # plt.show()
                #
                # plt.imshow(imgs[49, :, :, 0])
                # plt.show()




                # ---------------------
                #  Train Discriminator
                # ---------------------


                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)



                # -----------------
                #  Train Generator
                # -----------------

                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, valid)




                # Plot the progress

                elapsed_time = datetime.datetime.now() - start_time

                print("%d [D loss_real: %f, loss_fake: %f, acc_real: %.2f%%, acc_fake: %.2f%%] [G loss: %f] time: %s" % (
                    epoch,
                    d_loss_real[0],
                    d_loss_fake[0],
                    100 * d_loss_real[1],
                    100 * d_loss_fake[1],
                    g_loss,
                    elapsed_time))

                # If at save interval => save generated image samples
                if (epoch % sample_interval == 0) and (batch_i == 0):
                    self.sample_images(epoch)


    def sample_images(self, epoch):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    gan = MuiltiChannelGAN()
    gan.train()

    noise = np.random.normal(0, 1, (30, 100))


    pred_img = gan.generator.predict_on_batch(noise)

    slice_idx = 5

    # plt.imshow(pred_img[slice_idx, :, :, 0])
    # plt.show()