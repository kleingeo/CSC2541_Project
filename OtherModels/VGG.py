'''
2019-03-02
author: Phil Boyer
Simple implementation of VGG as in 2014 paper (with no opportunity to significantly change architecture)
'''

from keras.layers import Input, MaxPooling2D, Conv2D, BatchNormalization, \
    Activation, Deconvolution2D, UpSampling2D, concatenate, Dropout, Dense, \
    ZeroPadding2D, Reshape, Permute
from keras.activations import softmax
from keras.models import Model
from keras import backend as K
from keras import regularizers
K.set_image_dim_ordering('tf')

def get_vgg(img_x = 256,
            img_y = 256,
            dropout = 0,
            num_seq = 1):


    model_inputs = Input((img_x,
                          img_y,
                          num_seq))

    #kernel_size = (kernel_1d_size, kernel_1d_size)
    #dilation_rate = (dilation_rate, dilation_rate)
    #pool_size = (pool_1d_size, pool_1d_size)
    #current_layer = model_inputs
    #levels = list()


    #inputs are 256x256 greyscale images

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(model_inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2, padding='same')(x)
    f1 = x

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2, padding='same')(x)
    f2 = x

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2, padding='same')(x)
    f3 = x

    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2, padding='same')(x)
    f4 = x

    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=2, padding='same')(x)
    f5 = x

    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(5e-4))(x)
    x = Dropout(dropout)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(5e-4))(x)
    x = Dropout(dropout)(x)
    x = Dense(1000, activation='relu')(x)
    x = Activation(softmax)(x)

    levels = [f1, f2, f3, f4, f5]

    o = levels[2]

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(1, (3, 3), padding='same')(o)
    o = (Activation('softmax'))(o)
    model = Model(model_inputs, o)

    return model


if __name__ == '__main__':
    a = get_vgg()


    print('done')
