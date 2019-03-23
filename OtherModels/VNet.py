'''
2019-03-18
author: Geoff Klein
edited: G. Kuling
Simple implementation of VNet
'''
import numpy as np
from keras import backend as K
import keras.layers as KL
from keras.layers import Activation, Deconvolution2D
from keras.layers import Conv2D, UpSampling2D, \
    BatchNormalization, Dropout
from keras.layers import Input, concatenate
from keras.models import Model

K.set_image_dim_ordering('tf')

def get_vnet(
        num_channels=1,
        img_shape=(256, 256),
        num_classes=1,
        dilation_rate=1,
        kernel_initializer='glorot_uniform',
        kernel_1d_size=3,
        depth=5,
        number_of_base_filters=16,
        batch_normalization=False,
        pool_size=(2, 2),
        deconvolution=True,
        dropout=0):
    """
    mostly taken from
    https://raw.githubusercontent.com/ellisdg/3DUnetCNN/master/unet3d/model/unet.py
    :param number_of_sequences: this is 1 if only one sequence is used in the
    batches, and  >1 if more sequences has been used.
    :param volume_frames: number of slices in the volume
    :param volume_rows: the number of rows in the volume
    :param volume_columns: the number of coloumns in the volume
    :param input_optimizer: optimizer of the model
    :param dilation_rate: the dilation rate of the convolutuinal filters
    :param kernel_initializer:
    :param kernel_1d_size: the size of the kernel, if it is 3 , then the kernel
    is (3,3,3)
    :param depth: the number of blocks in U-Net
    :param number_of_base_filters: number of filters in the first convolutional
    layer of the U-Net, the number will be multiplied by 2 in the blocks
    moving down
    :param batch_normalization: bool value, True means the batch normalization
    layers are added to the blocks, false, means there is no batch normalization
    blocks
    :param pool_size:
    :param deconvolution: a bool value, true deconvolution layer is used, false
    up sampling is used
    :param dropout: a bool value, true means the dropout layer is in place,
    false means the drop out is not used
    :return:
    """


    model_inputs = Input((img_shape + (num_channels,)))


    kernel_size = (kernel_1d_size, kernel_1d_size)
    dilation_rate = (dilation_rate, dilation_rate)
    current_layer = model_inputs
    levels = list()

    for layer_depth in range(depth):


        block1 = create_convolution_block(
            input_layer=current_layer,
            n_filters=number_of_base_filters * (2 ** layer_depth),
            batch_normalization=batch_normalization,
            kernel=kernel_size,
            kernel_initializer=kernel_initializer,
            dilation_rate=dilation_rate,
            depth=str(layer_depth))

        final_layer_block = block1

        if layer_depth > 0:

            block2 = create_convolution_block(
                input_layer=block1,
                n_filters=number_of_base_filters * (2 ** layer_depth),
                batch_normalization=batch_normalization,
                kernel=kernel_size,
                kernel_initializer=kernel_initializer,
                dilation_rate=dilation_rate,
                depth=str(layer_depth) + '_2')

            final_layer_block = block2

        if layer_depth > 1:

            block3 = create_convolution_block(
                input_layer=block2,
                n_filters=number_of_base_filters * (2 ** layer_depth),
                batch_normalization=batch_normalization,
                kernel=kernel_size,
                kernel_initializer=kernel_initializer,
                dilation_rate=dilation_rate,
                depth=str(layer_depth) + '_3')

            final_layer_block = block3


        shortcut = KL.Conv2D(number_of_base_filters * (2 ** layer_depth), (1, 1))(current_layer)

        skip_block = KL.Add()([final_layer_block, shortcut])



        if layer_depth < depth - 1:
            current_layer = KL.Conv2D(number_of_base_filters * (2 ** layer_depth),
                                      (2, 2),
                                      strides=2)(skip_block)
            current_layer = KL.PReLU()(current_layer)

            levels.append(skip_block)

        else:
            current_layer = skip_block


    # going up
    for layer_depth in range(depth - 2, -1, -1):

        up_convolution = get_up_convolution(input=current_layer,
                                            pool_size=pool_size,
                                            deconvolution=deconvolution,
                                            n_filters=number_of_base_filters * (2 ** layer_depth),
                                            depth='UP' + str(layer_depth))

        concat = concatenate([up_convolution, levels[layer_depth]], axis=-1)

        if dropout > 0:
            concat = Dropout(dropout)(concat)

        current_layer = create_convolution_block(
            n_filters=number_of_base_filters * (2 ** layer_depth) * 2,
            input_layer=concat,
            batch_normalization=batch_normalization,
            kernel=kernel_size,
            kernel_initializer=kernel_initializer,
            dilation_rate=dilation_rate,
            depth='UP' + str(layer_depth))


        if layer_depth > 0:

            current_layer = create_convolution_block(
                n_filters=number_of_base_filters * (2 ** layer_depth) * 2,
                input_layer=current_layer,
                batch_normalization=batch_normalization,
                kernel=kernel_size,
                kernel_initializer=kernel_initializer,
                dilation_rate=dilation_rate,
                depth='UP' + str(layer_depth) + '_2')


        if layer_depth > 1:

            current_layer = create_convolution_block(
                n_filters=number_of_base_filters * (2 ** layer_depth) * 2,
                input_layer=current_layer,
                batch_normalization=batch_normalization,
                kernel=kernel_size,
                kernel_initializer=kernel_initializer,
                dilation_rate=dilation_rate,
                depth='UP' + str(layer_depth) + '_3')



        shortcut = KL.Conv2D(number_of_base_filters * (2 ** layer_depth) * 2,
                             (1, 1))(up_convolution)

        skip_block = KL.Add()([current_layer, shortcut])

    n_labels = 1
    final_convolution = Conv2D(num_classes, (1, 1))(skip_block)
    act = Activation('sigmoid')(final_convolution)

    model = Model(inputs=model_inputs, outputs=act)
    return model

    # model.compile(optimizer=input_optimizer, loss=dice_coef_loss,
    #               metrics=[concurrency, dice_coef])
    # print(model.summary())


    # if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
    #     CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    # else:
    #     CUDA_VISIBLE_DEVICES = ['None']
    #
    # if multi_gpu == True and len(CUDA_VISIBLE_DEVICES) > 1:
    #     from keras.losses import binary_crossentropy
    #     model_parallel = multi_gpu_model(model, gpus=len(CUDA_VISIBLE_DEVICES))
    #
    #     model_parallel.compile(optimizer=input_optimizer, loss=dice_coef_loss,
    #                            metrics=[concurrency, dice_coef])
    #
    #     return model, model_parallel
    #
    # else:
    #     from keras.losses import binary_crossentropy
    #     model.compile(optimizer=input_optimizer, loss=dice_coef_loss,
    #                   metrics=[concurrency, dice_coef])
    #
    #     model_parallel = model
    #     return model, model_parallel


    # if multi_gpu == True:
    #
    #     model_parallel = multi_gpu_model(model)
    #
    #     model_parallel.compile(optimizer=input_optimizer, loss=dice_coef_loss,
    #                            metrics=[concurrency, dice_coef])
    #
    #     return model, model_parallel
    #
    # else:
    #
    #     model.compile(optimizer=input_optimizer, loss=dice_coef_loss,
    #                   metrics=[concurrency, dice_coef])
    #
    #     model_parallel = None
    #     return model, model_parallel


def create_convolution_block(input_layer,
                             n_filters,
                             batch_normalization=False,
                             kernel=(3, 3),
                             activation=None,
                             padding='same',
                             strides=(1, 1),
                             instance_normalization=False,
                             dilation_rate=(1, 1),
                             kernel_initializer='glorot_uniform',
                             depth=None,
                             kern_reg=None,
                             bias_reg=None):
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
                   kernel_initializer=kernel_initializer,
                   name='conv{}'.format(depth))(input_layer)

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


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    from https://raw.githubusercontent.com/ellisdg/3DUnetCNN/
    master/unet3d/model/unet.py

    Each level has a particular output shape based on the number of filters
    used in that level and the depth or number
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model
    a given node is.
    :return: 5D vector of the shape of the output node
    """
    output_image_shape = np.asarray(
        np.divide(image_shape, np.power(pool_size, depth)),
        dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(input,
                       n_filters,
                       pool_size,
                       kernel_size=(2, 2),
                       strides=(2, 2),
                       deconvolution=False,
                       depth=None):
    if deconvolution:
        return Deconvolution2D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides,
                               name=depth)(input)
    else:
        return Conv2D(filters=n_filters, kernel_size=kernel_size,
                      name=depth, padding='same')(UpSampling2D(
            size=pool_size)(input))


