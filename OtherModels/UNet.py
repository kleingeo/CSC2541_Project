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
K.set_image_dim_ordering('tf')

def get_unet(img_shape = (512, 512),
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
             num_classes = 3,
             num_channels = 1):


    model_inputs = Input((img_shape + (num_channels,)))

    kernel_size = (kernel_1d_size, kernel_1d_size)
    dilation_rate = (dilation_rate, dilation_rate)
    pool_size = (pool_1d_size, pool_1d_size)
    current_layer = model_inputs
    levels = list()

    ### create Downsampling Arm
    for layer_depth in range(depth):
        layer1 = create_convolution_block(
            input_layer=current_layer,
            n_filters=base_filter * (2 ** layer_depth),
            batch_normalization=batch_normalization)
        layer2 = create_convolution_block(
            input_layer=layer1,
            n_filters=base_filter * (2 ** layer_depth),
            batch_normalization=batch_normalization,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer
        )

        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

     ### create Upsample Arm
    for layer_depth in range(depth - 2, -1, -1):
        up_convolution = get_up_convolution(
            pool_size=pool_1d_size,
            deconvolution=deconvolution,
            n_filters=current_layer._keras_shape[1])(current_layer)

        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=-1)

        if dropout > 0:
            concat = Dropout(dropout)(concat)

        current_layer = create_convolution_block(
            n_filters=levels[layer_depth][1]._keras_shape[1],
            input_layer=concat,
            batch_normalization=batch_normalization,
            kernel=kernel_size,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer)

        current_layer = create_convolution_block(
            n_filters=levels[layer_depth][1]._keras_shape[1],
            input_layer=current_layer,
            batch_normalization=batch_normalization,
            kernel=kernel_size,
            kernel_initializer=kernel_initializer,
            dilation_rate=dilation_rate)

    ### Finish off the Output and Optimize
    n_labels = num_classes
    final_convolution = Conv2D(n_labels, (1, 1))(current_layer)
    act = Activation("sigmoid")(final_convolution)
    model = Model(inputs=model_inputs, outputs=act)

    # model.compile(optimizer=optimizer, loss=dice_loss,
    #               metrics=[dice_coef])
    # print(model.summary())
    return model

def create_convolution_block(input_layer,
                             n_filters,
                             batch_normalization=False,
                             kernel=( 3, 3),
                             activation=None,
                             padding='same',
                             strides=( 1, 1),
                             instance_normalization=False,
                             dilation_rate=( 1, 1),
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
                   dilation_rate=dilation_rate,
                   padding=padding,
                   strides=strides,
                   kernel_initializer=kernel_initializer)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
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

def get_up_convolution(n_filters,
                       pool_size,
                       kernel_size=(2, 2),
                       strides=(2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution2D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling2D(size=pool_size)

if __name__ == '__main__':
    a = get_unet()


    print('done')