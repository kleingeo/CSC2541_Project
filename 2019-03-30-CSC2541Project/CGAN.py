'''
2019-03-31
Codes adapted from https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py

'''

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import model_from_json
import DGenUtils as Utils
import TrainerFileNamesUtil as TrainerFileNamesUtil
from keras import backend as K

import os

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
K.set_image_dim_ordering('th')

class CGAN():
    def __init__(self,
                 pretrained_folder,
                 data_dir,
                 target_dir,
                 ofolder):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.channels, self.img_rows, self.img_cols )
        self.input_shape = (2, self.img_rows, self.img_cols)
        # self.latent_dim = 100
        self.pretrained_folder = pretrained_folder
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.ofolder = ofolder

        optimizer = Adam(1e-5, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.input_shape))
        img = self.generator([noise])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(noise, valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):

        json_file_name = \
            [i for i in os.listdir(self.pretrained_folder) if i.endswith('json')][0]
        weights_file_name = \
            [i for i in os.listdir(self.pretrained_folder) if i.startswith(
                'model_best')][0]
        json_file = open(''.join([self.pretrained_folder, '/', json_file_name]))
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # Load the weights to the model
        model.load_weights(''.join([self.pretrained_folder, '/', weights_file_name]))

        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(8, kernel_size=4, strides=2, padding='same',
                         input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.8))
        model.add(Conv2D(16, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(32, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        # model.summary()

        # img = Input(shape=self.img_shape)
        # label = Input(shape=(1,), dtype='int32')
        #
        # # label_embedding = Flatten()(Embedding(np.prod(self.img_shape))(label))
        # flat_img = Flatten()(img)
        #
        # model_input = multiply([flat_img])
        #
        # validity = model(model_input)
        #
        # return Model([img, label], validity)
        return model

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        data_files = [''.join([self.data_dir, i]) for i in
                           os.listdir(self.data_dir)]
        target_files = [''.join([self.target_dir, i]) for i in
                             os.listdir(self.target_dir)]
        assert len(data_files) == len(target_files), \
            'Sample list and target list are different sizes.'

        r_training_data, r_target_data = Utils.PreprocessTrainingData(
            data_files, target_files, (128,128))

        # Build atlas for the augmentor UNet
        atlas = np.zeros((128,128))

        for i in range(len(r_target_data)):
            atlas = atlas + r_target_data[i]

        atlas = np.divide(atlas, len(r_target_data))
        n_sam = len(r_training_data)
        thrs = np.random.random((n_sam))
        f_input_data =[]
        for i in range(n_sam):
            map = np.zeros(( 2, 128, 128))
            map[ 1, :, :] = (atlas > thrs[i]).astype('uint8')
            map[ 0, :, :] = np.ones((128, 128)) - (atlas > thrs[i]).astype(
                'uint8')

            f_input_data.append(map)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, len(r_training_data), batch_size)
            imgs = np.asarray([np.expand_dims(r_training_data[k], axis=0) for k\
                    in idx])

            # Sample noise as generator input
            noise = np.asarray([f_input_data[k] for k in idx])

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels


            # Train the generator
            g_loss = self.combined.train_on_batch([noise], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, f_input_data)

        ## Save the model after training
        model_json = self.generator.to_json()
        with open(TrainerFileNamesUtil.create_model_json_filename(
                output_directory=self.ofolder,
                time_stamp=TrainerFileNamesUtil.get_time(),
                keyword='UNet_cGANAugmentor'), 'w') as \
                jason_file:
            jason_file.write(model_json)
        self.generator.save(self.ofolder + '/model_best_weights_.h5')

    def sample_images(self, epoch, f_input_data):
        r, c = 2, 2
        idx = np.random.randint(0, len(f_input_data), r*c)

        noise = np.asarray([f_input_data[k] for k in idx])

        gen_imgs = self.generator.predict([noise])

        # Rescale images 0 - 255
        gen_imgs = (0.5 * gen_imgs + 0.5)*255

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,0,:,:], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(self.ofolder + "/images_epoch_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cgan = CGAN(pretrained_folder=r'X:\2019-03-30-CSC2541Project'
                                  r'\UNetAugmentor\\',
                 data_dir=r'D:\prostate_data\Task05_Prostate\imagesTr\\',
                 target_dir=r'D:\prostate_data\Task05_Prostate\labelsTr\\',
                 ofolder=r'X:\2019-03-30-CSC2541Project\cGANUnetAugmentor\\')
    cgan.train(epochs=1, batch_size=50, sample_interval=1)

    print('done')