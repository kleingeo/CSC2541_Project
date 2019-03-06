"""
(C) Martellab, 2018

@author: homa + Grey
"""
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.ndimage import morphology


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


def surface_distance(input1, input2, sampling=1, connectivity=1):
    """
     calculates surface distances of 2 binary objects.
     Found at https://mlnotebook.github.io/post/surface-distance-function/

    :param input1: volume one
    :param input2: volume two
    :param sampling:
    :param connectivity:
    :return:
    """

    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 - morphology.binary_erosion(input_1, conn)
    Sprime = input_2 - morphology.binary_erosion(input_2, conn)

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return sds


def mean_surface_distance(y_true, y_pred):
    """
    This method calculates the mean surface distance
    :param y_true: numpy array
    :param y_pred: numpy array
    :return: mean surface distance
    """

    return np.mean(surface_distance(y_pred, y_true))


def relative_absolute_volume_difference(y_true, y_pred):
    """
    This will calculate the relative absolute volume difference
    :param y_true: numpy array
    :param y_pred: numpy array
    :return: relative absolute volume difference
    """
    if y_true.max() != 1 and y_true.max() > 0:
        y_true = y_true / y_true.max()
    if y_pred.max() != 1 and y_pred.max() > 0:
        y_pred = y_pred / y_pred.max()

    vol_pred = np.sum(y_pred)
    vol_true = np.sum(y_true)
    if vol_true == 0:
        vol_true = np.finfo(float).eps

    return np.abs(vol_pred - vol_true) / vol_true


def jaccard_coefficient(y_true, y_pred):
    """
    This function returns the Jaccard coefficient
    :param y_true: numpy array
    :param y_pred: numpy array
    :return: jaccard coefficient
    """
    epsilon = np.finfo(float).eps
    y_true = np.asarray(y_true).astype(np.bool)
    y_pred = np.asarray(y_pred).astype(np.bool)

    if y_pred.shape != y_true.shape:
        raise ValueError("shape mismatch in dice coefficient calculations.")

    intersection = np.logical_and(y_true, y_pred)
    jaccard = (intersection.sum() + epsilon) / \
              (y_true.sum() + y_pred.sum() - intersection.sum() + epsilon)
    return jaccard


def accuracy_metric(y_true, y_pred):
    """
    This function returns the accuracy
    :param y_true: numpy array
    :param y_pred: numpy array
    :return: accuracy
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("shape mismatch in dice coefficient calculations.")

    y_true = np.asarray(y_true).astype(np.bool)
    y_pred = np.asarray(y_pred).astype(np.bool)

    tn, fp, fn, tp = \
        confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()

    denominator = tp + fp + fn + tn
    if denominator == 0:
        denominator = np.finfo(float).eps
    accuracy = (tp + tn) / denominator
    return accuracy


def sensitivity_metric(y_true, y_pred):
    """
    This function returns the sensitivity
    :param y_true: numpy array
    :param y_pred: numpy array
    :return: accuracy
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("shape mismatch in dice coefficient calculations.")
    y_true = np.asarray(y_true).astype(np.bool)
    y_pred = np.asarray(y_pred).astype(np.bool)

    tn, fp, fn, tp = \
        confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()
    denominator = tp + fp
    if denominator == 0:
        denominator = np.finfo(float).eps

    sensitivity = tp / denominator
    return sensitivity


def specificity_metric(y_true, y_pred):
    """
    This function returns the specificity
    :param y_true: numpy array
    :param y_pred: numpy array
    :return: accuracy
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("shape mismatch in dice coefficient calculations.")
    y_true = np.asarray(y_true).astype(np.bool)
    y_pred = np.asarray(y_pred).astype(np.bool)

    tn, fp, fn, tp = \
        confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()
    denominator = fn + tn

    if denominator == 0:
        denominator = np.finfo(float).eps
    specificity = tn / denominator

    return specificity


def false_negative_rate(y_true, y_pred):
    """
    This function returns the false negative rate
    :param y_true: numpy array
    :param y_pred: numpy array
    :return: accuracy
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("shape mismatch in dice coefficient calculations.")

    y_true = np.asarray(y_true).astype(np.bool)
    y_pred = np.asarray(y_pred).astype(np.bool)

    tn, fp, fn, tp = \
        confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()
    denominator = tp + fn
    if denominator == 0:
        denominator = np.finfo(float).eps
    false_negative = fn / denominator

    return false_negative


def false_positive_rate(y_true, y_pred):
    """
    This function returns the False Positive Rate
    :param y_true: numpy array
    :param y_pred: numpy array
    :return: accuracy
    """
    if y_pred.shape != y_true.shape:
        raise ValueError("shape mismatch in dice coefficient calculations.")
    y_true = np.asarray(y_true).astype(np.bool)
    y_pred = np.asarray(y_pred).astype(np.bool)

    tn, fp, fn, tp = \
        confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()
    denominator = fp + tn
    if denominator == 0:
        denominator = np.finfo(float).eps
    false_positive = fp / denominator

    return false_positive
