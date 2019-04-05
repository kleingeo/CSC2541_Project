import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter



def random_scaling(scaling):

    sx = np.random.uniform(scaling[0], scaling[1])
    sy = np.random.uniform(scaling[0], scaling[1])

    scaling_mat = np.array([[sx, 0, 1],
                            [0, sy, 1],
                            [0, 0, 1]])

    return scaling_mat


def random_translate(scaling, sample_size):

    translate = np.array([0, 0])


    sx = np.random.uniform(scaling[0], scaling[1])
    sy = np.random.uniform(scaling[0], scaling[1])


    for i, s in enumerate([sx, sy]):
        if s > 1:
            translate[i] = np.random.uniform(-sample_size[i] * (s - 1), 0)
        else:
            translate[i] = np.random.uniform(0, sample_size[i] * (1 - s))

    shift_matrix = np.eye(3)
    shift_matrix[:2, 2] = translate

    return shift_matrix


def random_rotation(rotation):


    theta = np.random.uniform(-rotation, rotation)

    sin = np.sin(theta)
    cos = np.cos(theta)

    rot = np.array([[cos, sin, 0],
                    [sin, -cos, 0],
                    [0, 0, 1]])

    return rot


def random_shear(shearing):

    hx = np.random.uniform(-shearing, shearing)
    hy = np.random.uniform(-shearing, shearing)

    shear_mat = np.array([[1, hy, 0],
                          [hx, 1, 0],
                          [0, 0, 1]])

    return shear_mat


def determine_operations(rescaling, translation, rotation, sheering, elastic,
                         lr_flip_slice, ud_flip_slice):

    if rescaling is not None:
        rescaling = np.random.random_integers(0, 1)
    else:
        rescaling = 0


    if translation is not None:
        translation = np.random.random_integers(0, 1)
    else:
        translation = 0


    if rotation is not None:
        rotation = np.random.random_integers(0, 1)
    else:
        rotation = 0



    if sheering is not None:
        sheering = np.random.random_integers(0, 1)
    else:
        sheering = 0

    if elastic is not None:
        elastic = np.random.random_integers(0, 1)
    else:
        elastic = 0


    if lr_flip_slice is not False:
        lr_flip_slice = np.random.random_integers(0, 1)
    else:
        lr_flip_slice = 0


    if ud_flip_slice is not False:
        ud_flip_slice = np.random.random_integers(0, 1)
    else:
        ud_flip_slice = 0


    return (rescaling, translation, rotation, sheering, elastic,
            lr_flip_slice, ud_flip_slice)


def elastic_transform(image, alpha, sigma, order, random_state=None):
    '''
    Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    :param image: Input 2D image
    :param alpha: Scaling factor. Controls intensity of the deformation.
    :param sigma: Standard deviation of gaussian filter. Functionally the elasticity coefficient.
    :return:
    '''

    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(image, indices, order=order).reshape(shape)


def RandomAugmentation(img, seg_img, sample_size=(256, 256),
                       scaling=None, translation=None, rotation=None, shearing=None, elastic=None,
                       lr_flip_slice=False, ud_flip_slice=False):

    """
    :param img: Input sample CT spine image
    :param seg_img: Input sample segmentation
    :param spacing:
    :param sample_size: Size of the samples in [x, y, z]
    :param univoxal:
    :param scaling: Parameters used to determine the range the sample is scaled for augmentation. Can be either a list
                    of length 2 or length 6. Length 2 specifies the lower and upper bounds for isotropic scaling in
                    x, y and z. Length 6 specifies the low and upper bounds for the specific x, y and z scaling
                    (ie, [xyz_lower, xyz_upper] or [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper])
    :param translation: Specifies if the image is translated in the augmentation. Translations are based around the
                        scaling bounds.
    :param rotation: Specifies if the image is rotate. Can be either a list of length 2 or length 6. Length 2 specifies
                     the lower and upper bounds for rotation only about the z-axis. Length 6 specifies the low and upper
                     bounds for a complete Euler rotation matrix.
                     (ie, [z_lower, z_upper] or [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper])
    :param shearing: Parameter used to decide if the image is sheared. Can either be a list of length 2 or length 6.
                     Length 2 specifies the lower and upper bounds for shearing along the x-direction only, whereas a
                     length of 6 specifies the lower and upper bounds for shearing along x, y and z.
                     (ie, [x_lower, x_upper] or [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper])
    :param lr_flip_slice: Boolean to decide if the axial (transverse) planes of the image should be flipped along the
                          longitudinal direction.
                          (ie, flip each slice along the z-axis in the y-axis, so x indices are reversed)
    :param ud_flip_slice: Boolean to decide if the axial (transverse) planes of the image should be flipped along the
                          lateral direction.
                          (ie, flip each slice along the z-axis in the x-axis, so y indices are reversed)
    :return:
    """



    (scaling_prob, translation_prob, rotation_prob, shearing_prob, elastic_prob,
     lr_flip_slice_prob, ud_flip_slice_prob) = determine_operations(scaling, translation, rotation, shearing, elastic,
                                                                    lr_flip_slice, ud_flip_slice)

    img_data = img
    seg_data = seg_img


    transform_matrix = None

    if rotation_prob:
        rot = random_rotation(rotation)

        transform_matrix = rot

    if scaling_prob:

        scaling_mat = random_scaling(scaling)

        transform_matrix = scaling_mat if transform_matrix is None else np.dot(transform_matrix, scaling_mat)


    if translation_prob:

        shift_matrix = random_translate(scaling, sample_size)

        transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)


    if shearing_prob:

        shear_mat = random_shear(shearing)

        transform_matrix = shear_mat if transform_matrix is None else np.dot(transform_matrix, shear_mat)


    if transform_matrix is not None:

        o_x = float(sample_size[0]) / 2 + 0.5
        o_y = float(sample_size[1]) / 2 + 0.5

        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])


        transform_matrix = np.dot(np.dot(offset_matrix, transform_matrix), reset_matrix)


        # img_data = ndi.interpolation.affine_transform(img_data, transform_matrix, order=1)
        # seg_data = ndi.interpolation.affine_transform(seg_data, transform_matrix, order=0)

        img_data = ndi.interpolation.affine_transform(img_data, matrix=transform_matrix[:2, :2],
                                                      offset=transform_matrix[:2, -1], order=1)

        seg_data = ndi.interpolation.affine_transform(seg_data, matrix=transform_matrix[:2, :2],
                                                      offset=transform_matrix[:2, -1], order=0)


        if elastic_prob:
            img_data = elastic_transform(img_data, alpha=720, sigma=24, order=1)
            seg_data = elastic_transform(seg_data, alpha=720, sigma=24, order=0)


    if lr_flip_slice_prob:
        img_data = img_data[::-1, :]
        seg_data = seg_data[::-1, :]

    if ud_flip_slice_prob:
        img_data = img_data[:, ::-1]
        seg_data = seg_data[:, ::-1]


    img_data = img_data.astype(np.float32)
    seg_data = seg_data.astype(np.uint8)



    return img_data, seg_data
