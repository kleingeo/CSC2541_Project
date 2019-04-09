from keras import backend as K
import numpy as np

def dice_coef(y_true, y_pred):
    """This method calculates the dice coefficient between the true
     and predicted masks
    Args:
        y_true: The true mask(i.e. ground-truth or expert annotated mask)
        y_pred: The predicted mask

    Returns:
        double: The dice coefficient"""
    smooth = K.epsilon()
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f)
                                           + smooth)

def dice_loss(y_true, y_pred):
    '''
    dice coefficient loss function.
    :param y_true: The true mask(i.e. ground-truth or expert annotated mask)
    :param y_pred: The predicted mask
    :return: negative of the dice_loss
    '''
    return -dice_coef(y_true, y_pred)

def dice_coefficient_numpy_arrays(y_true, y_pred):
    """
    this method calculates the dice coefficient, there is no dependency
    on tensorflow,
    :param y_true: numpy array
    :param y_pred: numpy array
    :return: dice coefficient
    """
    epsilon = np.finfo(float).eps
    y_true = np.asarray(y_true).astype(np.bool)
    y_pred = np.asarray(y_pred).astype(np.bool)

    if y_pred.shape != y_true.shape:
        raise ValueError("shape mismatch in dice coefficient calculations.")
    intersection = np.logical_and(y_true, y_pred)
    dice_value = \
        (2.0 * intersection.sum() + epsilon) / \
        (y_true.sum() + y_pred.sum() + epsilon)
    return dice_value

def My_new_loss(y_true, y_pred):
    '''
    My Troversky loss function.
    :param y_true:
    :param y_pred:
    :return:
    '''

    ave_DSC = (dice_coef(y_true[:, 0], y_pred[:, 0]) +
               dice_coef(y_true[:,1] ,y_pred[:,1]))

    return 2.0 - ave_DSC
