'''
2019-03-02
author: Phil Boyer
Simple implementation of VGG as in 2014 paper (with no opportunity to significantly change architecture)
'''
from Utils import decide_chnls
from keras.layers import Input, MaxPooling2D, Conv2D, BatchNormalization, \
    Activation, Deconvolution2D, UpSampling2D, concatenate, Dropout, Dense
from keras.models import Model
from keras import backend as K
from keras import regularizers
K.set_image_dim_ordering('th')

def get_vgg(mode = '2Ch',
             img_x = 512,
             img_y = 512,
             optimizer = 'ADAM',
             dilation_rate = 1,
             kernel_initializer = 'glorot_uniform',
             kernel_1d_size = 3,
             depth = 5,
             base_filter = 16,
             batch_normalization = False,
             pool_1d_size = 2,
             deconvolution = False,
             dropout = 0,
             num_classes = 3):

    num_seq, _ = decide_chnls(mode) #???? How many channels?

    model_inputs = Input((num_seq,
                          img_x,
                          img_y))

    #kernel_size = (kernel_1d_size, kernel_1d_size)
    #dilation_rate = (dilation_rate, dilation_rate)
    #pool_size = (pool_1d_size, pool_1d_size)
    #current_layer = model_inputs
    #levels = list()


    #inputs are 256x256 greyscale images

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(model_inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2, padding='same')(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2, padding='same')(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2, padding='same')(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2, padding='same')(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2, padding='same')(x)

    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(5e-4), dropout=0.5)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(5e-4), dropout=0.5)(x)
    x = Dense(1000, activation='relu')(x)
    x = Activation('soft_max')(x)

    print(x.summary())

    return Model(model_inputs, x)


if __name__ == '__main__':
    a = get_vgg()


    print('done')
