'''
2018-09-28
(c) A. Martel Lab Co.
author: G.Kuling
This is my Inception UNet code. When get_iunet is called, it reutrns a 2D
inception unet with the given parameters.
'''
from keras.layers import Input, MaxPooling2D, Conv2D, BatchNormalization, \
    Activation, Deconvolution2D, UpSampling2D, concatenate, Dropout
from keras.models import Model
from keras import backend as K
K.set_image_dim_ordering('tf')

def get_iunet(img_x = 512,
              img_y = 512,
              dilation_rate = 1,
              kernel_initializer = 'glorot_uniform',
              depth = 5,
              base_filter = 16,
              batch_normalization = False,
              pool_1d_size = 2,
              deconvolution = False,
              dropout = 0,
              num_classes = 1,
              num_seq = 1):
    """
    The function that build the Inception Unet
    :param mode: (str) the choice of imaging modality. '2Ch', 'WOFS' or 'FS'
    :param img_x: (int) image input x size
    :param img_y: (int) image input y size
    :param optimizer: optimizer to be used
    :param dilation_rate: (int) the spacial separation of filters.
    :param kernel_initializer: (str) the initialzation function used for the
    filter weights
    :param depth: (int) how many steps down the unet will go.
    :param base_filter: (int) the amount of output filters for each depth of
    the unet.
    :param batch_normalization: (bool) option for using batch normalization
    :param pool_1d_size: (int) the pool filter size in 1 direction
    :param deconvolution: (bool) choice of upsampling or deconvolutional
    filters.
    :param dropout: (float) the level of dropout used in the dropout layers
    :param num_classes: (int) the amount of output masks needed.
    :return: an Inception UNet
    """

    model_inputs = Input((img_x,
                          img_y,
                          num_seq))

    dilation_rate = (dilation_rate, dilation_rate)
    pool_size = (pool_1d_size, pool_1d_size)
    current_layer = model_inputs
    levels = list()

    ### create Downsampling Arm
    for layer_depth in range(depth):
        layer1 = create_inception_block(
            input_layer=current_layer,
            n_filters=base_filter * (2 ** layer_depth),
            batch_normalization=batch_normalization,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer)

        if layer_depth < depth - 1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer1)
            levels.append([layer1,  current_layer])
        else:
            current_layer = layer1
            levels.append([layer1])

     ### create Upsample Arm
    for layer_depth in range(depth - 2, -1, -1):
        up_convolution = get_up_convolution(
            pool_size=pool_1d_size,
            deconvolution=deconvolution,
            n_filters=current_layer._keras_shape[1])(current_layer)

        concat = concatenate([up_convolution, levels[layer_depth][0]], axis=-1)

        if dropout > 0:
            concat = Dropout(dropout)(concat)

        current_layer = create_inception_block(
            n_filters=levels[layer_depth][1]._keras_shape[1],
            input_layer=concat,
            batch_normalization=batch_normalization,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer)


    ### Finish off the Output and Optimize
    final_convolution = Conv2D(num_classes, (1, 1))(current_layer)
    act = Activation("sigmoid")(final_convolution)
    model = Model(inputs=model_inputs, outputs=act)

    print(model.summary())
    return model

def create_inception_block(input_layer,
                           n_filters,
                           batch_normalization=False,
                           activation=None,
                           instance_normalization=False,
                           dilation_rate=( 1, 1),
                           kernel_initializer='glorot_uniform'):
    """
    Creates an inception block
    :param input_layer: the input layer to continue from
    :param n_filters: the amount of output filters you desire.
    :param batch_normalization: Option for batch normalization
    :param activation: Activation function at the end. default is ReLu
    :param instance_normalization: option for instance normalization
    :param dilation_rate: the spatial spread of the filters
    :param kernel_initializer: initialization of kernal weights option.
    :return: an Inception module
    """
    n_filters = int(n_filters/4)
    t1 = Conv2D(filters=n_filters,
                kernel_size=(1, 1),
                activation='relu',
                padding='same',
                kernel_initializer=kernel_initializer,
                dilation_rate=dilation_rate)(input_layer)
    t2 = Conv2D(filters=n_filters,
                kernel_size=(1, 1),
                activation='relu',
                padding='same',
                kernel_initializer=kernel_initializer,
                dilation_rate=dilation_rate)(input_layer)
    t2 = Conv2D(filters=n_filters,
                kernel_size=(3, 3),
                activation='relu',
                padding='same',
                kernel_initializer=kernel_initializer,
                dilation_rate=dilation_rate)(t2)
    t3 = Conv2D(filters=n_filters,
                kernel_size=(1, 1),
                activation='relu',
                padding='same',
                kernel_initializer=kernel_initializer,
                dilation_rate=dilation_rate)(input_layer)
    t3 = Conv2D(filters=n_filters,
                kernel_size=(5, 5),
                activation='relu',
                padding='same',
                kernel_initializer=kernel_initializer,
                dilation_rate=dilation_rate)(t3)
    t4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                      padding='same')(input_layer)
    t4 = Conv2D(filters=n_filters,
                kernel_size=(1, 1),
                activation='relu',
                padding='same',
                kernel_initializer=kernel_initializer,
                dilation_rate=dilation_rate)(t4)
    layer = concatenate([t1, t2, t3, t4], axis=-1)
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
    """
    Gives the up sampling of upconvolution layer
    :param n_filters: The amount of filters desired
    :param pool_size:
    :param kernel_size:
    :param strides:
    :param deconvolution:
    :return: the upsampling or deconvolution layer
    """
    if deconvolution:
        return Deconvolution2D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling2D(size=pool_size)

if __name__ == '__main__':
    a = get_iunet()


    print('done')