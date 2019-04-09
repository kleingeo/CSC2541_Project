'''
2018-09-28
(c) A. Martel Lab Co.
author: G.Kuling
This is my UNet code. When get_unet is called, it reutrns a 2D unet with the
given parameters.
'''

from keras.layers import Input, MaxPooling2D, Conv2D, BatchNormalization, \
    Activation, Deconvolution2D, UpSampling2D, concatenate, Dropout
from keras.models import Model
from keras import backend as K
import keras

# K.set_image_dim_ordering('th')
# K.set_image_data_format('channels_first')

def get_unet(img_x=128,
             img_y=128,
             dilation_rate=1,
             kernel_initializer='glorot_uniform',
             kernel_1d_size=3,
             depth=5,
             base_filter=64,
             batch_normalization=False,
             pool_1d_size=2,
             deconvolution=False,
             dropout=0,
             num_classes=1,
             num_seq=1,
             reg_const=None):


    # K.set_image_dim_ordering('th')
    # K.set_image_data_format('channels_first')

    model_inputs = Input((img_x, img_y, num_seq))

    kernel_size = (kernel_1d_size, kernel_1d_size)
    dilation_rate = (dilation_rate, dilation_rate)
    pool_size = (pool_1d_size, pool_1d_size)
    current_layer = model_inputs
    levels = list()



    if reg_const is not None:

        kernel_reg = keras.regularizers.l2(reg_const)
        bias_reg = keras.regularizers.l2(reg_const)

    else:
        kernel_reg = None
        bias_reg = None

    ### create Downsampling Arm
    for layer_depth in range(depth):

        layer1 = create_convolution_block(
            input_layer=current_layer,
            n_filters=base_filter * (2 ** layer_depth),
            kernel_reg=kernel_reg,
            bias_reg=bias_reg,
            batch_normalization=batch_normalization,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer)

        layer2 = create_convolution_block(
            input_layer=layer1,
            n_filters=base_filter * (2 ** layer_depth),
            kernel_reg=kernel_reg,
            bias_reg=bias_reg,
            batch_normalization=batch_normalization,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer)


        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

     ### create Upsample Arm
    for layer_depth in range(depth - 2, -1, -1):

        up_convolution = get_up_convolution(
            input=current_layer,
            batch_normalization=batch_normalization,
            pool_size=pool_size,
            deconvolution=deconvolution,
            n_filters=current_layer._keras_shape[-1],
            kernel_reg=kernel_reg, bias_reg=bias_reg)

        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=-1)

        if dropout > 0:
            concat = Dropout(dropout)(concat)

        current_layer = create_convolution_block(
            input_layer=concat,
            n_filters=levels[layer_depth][1]._keras_shape[-1],
            kernel_reg=kernel_reg,
            bias_reg=bias_reg,
            batch_normalization=batch_normalization,
            kernel=kernel_size,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer)

        current_layer = create_convolution_block(
            input_layer=current_layer,
            n_filters=levels[layer_depth][1]._keras_shape[-1],
            kernel_reg=kernel_reg,
            bias_reg=bias_reg,
            batch_normalization=batch_normalization,
            kernel=kernel_size,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer)

    ### Finish off the Output and Optimize
    final_convolution = Conv2D(num_classes, (1, 1),
                               kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(current_layer)
    act = Activation("sigmoid")(final_convolution)
    model = Model(inputs=model_inputs, outputs=act)

    # model.compile(optimizer=optimizer, loss=dice_loss,
    #               metrics=[dice_coef])
    print(model.summary())
    return model

def create_convolution_block(input_layer,
                             n_filters,
                             batch_normalization=False,
                             kernel_reg=None,
                             bias_reg=None,
                             kernel=(3, 3),
                             activation=None,
                             padding='same',
                             strides=(1, 1),
                             instance_normalization=False,
                             dilation_rate=(1, 1),
                             kernel_initializer='glorot_uniform'):
    """
     from https://raw.githubusercontent.com/ellisdg/3DUnetCNN/
     master/unet3d/model/unet.py
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation:
    :param padding:
    :param strides:
    :param instance_normalization:
    :param dilation_rate:
    :param kernel_initializer:
    :return:
    """
    layer = Conv2D(filters=n_filters,
                   kernel_size=kernel,
                   kernel_regularizer=kernel_reg,
                   bias_regularizer=bias_reg,
                   dilation_rate=dilation_rate,
                   padding=padding,
                   strides=strides,
                   kernel_initializer=kernel_initializer)(input_layer)

    if batch_normalization:
        layer = BatchNormalization()(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError(
                "Install keras_contrib in order to use instance normalization.")
        layer = InstanceNormalization(axis=1)(layer)

    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

def get_up_convolution(input,
                       n_filters,
                       pool_size,
                       kernel_size=(2, 2),
                       strides=(2, 2),
                       batch_normalization=False,
                       deconvolution=False,
                       kernel_reg=None,
                       bias_reg=None):

    if deconvolution:
        x = Deconvolution2D(filters=n_filters, kernel_size=kernel_size,
                            strides=strides, kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(input)
    else:
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same',
                   kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(UpSampling2D(size=pool_size)(input))

    if batch_normalization:
        x = BatchNormalization()(x)

    return x

if __name__ == '__main__':
    a = get_unet()


    print('done')
