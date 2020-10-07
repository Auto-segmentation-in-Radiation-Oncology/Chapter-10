import numpy as np
import medpy.metric.binary as mp
import tensorflow as tf
from tensorflow.keras import backend as K
from functools import wraps

__author__ = "Evan Porter"
__copyright__ = "Copyright 2018, Beaumont Health"
__credits__ = ["Evan Porter", "Thomas Guerrero"]
__maintainer__ = "Evan Porter"
__email__ = "evan.porter@beaumont.org"
__status__ = "Research"


def masked(fn: object) -> object:
    """[decorator to mask for missing data in ground truth]

    Args:
        fn (object): [loss function]

    Returns:
        object: [generalized loss function]
    """
    def masked_fn(y_true: tf.Tensor, y_pred: tf.Tensor, *args, **kwargs) -> fn:
        dims = len(K.shape(y_true))
        sums = K.sum(y_true, axis=range(dims-1))
        mask = K.cast(K.greater(sums, 0), 'float32')
        return fn(y_true*mask, y_pred*mask, *args, **kwargs)
    return masked_fn


def generalized(fn: object) -> object:
    """[decorator to compute generalized weights for loss function]

    Args:
        fn (object): [loss function]

    Returns:
        object: [generalized loss function]

    Notes:
        Computed with normalized weights to prevent overflow errors
    """
    def gen_fn(y_true: tf.Tensor, y_pred: tf.Tensor, *args, **kwargs) -> fn:
        dims = len(K.shape(y_true))
        sums = K.sum(y_true, range(dims-1))
        total = K.sum(y_true)
        weights = 1 / ((sums / total)**2 + K.epsilon())
        weights /= K.max(weights)
        return fn(y_pred, y_true, *args, weights=weights, **kwargs)
    return gen_fn


def soft_generalized(fn: object) -> object:
    """[decorator to compute a softer generalization function]

    Args:
        fn (object): [loss function]

    Returns:
        object: [softly generalized loss function]

    Notes:
        Logarithmic based as opposed to squared in the hard version
         this makes it better suited for instances when a class could
         have a membership of zero which would dominate in the hard
         version
        Computed with normalized weights to prevent overflow errors
    """
    def soft_gen_fn(y_true: tf.Tensor, y_pred: tf.Tensor, *args, **kwargs) -> fn:
        dims = len(K.shape(y_true))
        sums = K.sum(y_true, range(dims-1)) + K.epsilon()
        total = K.sum(y_true)
        weights = K.abs(K.log(sums / total))
        weights /= K.max(weights)
        return fn(y_pred, y_true, *args, weights=weights, **kwargs)
    return soft_gen_fn


@generalized
def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor,
              weights: tf.Tensor = None) -> tf.Tensor:
    """[computes the dice loss]

    Args:
        y_true (tf.Tensor): [ground truth tensor]
        y_pred (tf.Tensor): [prediction tensor]
        weights ([tf.Tensor], optional): [per-channel weighting]. Defaults to None.

    Returns:
        tf.Tensor: [loss value]
    """
    if weights is not None:
        dims = len(K.int_shape(y_true))
        numerator = 2. * K.sum(weights * K.sum(y_true * y_pred, range(dims-1)))
        denominator = K.sum(weights * K.sum(y_true + y_pred, range(dims-1)))
    else:
        numerator = 2. * K.sum(y_true * y_pred)
        denominator = K.sum(y_pred) + K.sum(y_true)
    return 1 - (numerator + K.epsilon()) / (denominator + K.epsilon())


def hausdorff_loss(percentile: float) -> object:
    """[given a percentile, returns the hausdorff loss function]

    Args:
        percentile (float): [specifies the hausdorff distance percentile]

    Returns:
        object: [hausdorff loss function]

    Notes:
        Compatiable with generalized and masking generators
        Hausdorff distance loss is more computationally expensive than
            other provided metrics

    References:
        Hausdorff distance calculation based upon MedPy:
            https://loli.github.io/medpy/_modules/medpy/metric/binary.html#hd
    """
    def _hd_percentile(result: np.ndarray, reference: np.ndarray,
                       voxelspacing: np.ndarray = None,
                       connectivity: int = 1) -> float:
        hd1 = mp.__surface_distances(
            result, reference, voxelspacing, connectivity)
        hd2 = mp.__surface_distances(
            reference, result, voxelspacing, connectivity)
        return np.percentile(np.hstack((hd1, hd2)), percentile)

    #@masked
    #@generalized
    def hd_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor,
                   weights: tf.Tensor = None) -> tf.Tensor:
        y_true_np = K.eval(y_true)
        y_pred_np = K.eval(y_pred)
        shape = K.eval(K.shape(y_true))
        max_hd = np.sqrt(np.sum(shape[1:3]**2))
        hds = []

        for batch in range(shape[0]):
            temp = []
            for ch_index in range(shape[-1]):
                t_ch = y_true_np[batch, ..., ch_index]
                p_ch = y_pred_np[batch, ..., ch_index]
                t_sum = np.sum(t_ch)
                p_sum = np.sum(p_ch)
                if t_sum and p_sum:  # true positive
                    hd_np = _hd_percentile(t_ch, p_ch)
                elif not t_sum and p_sum:  # false positive
                    hd_np = max_hd
                elif t_sum and not p_sum:  # false negative
                    hd_np = max_hd
                else:  # true negative
                    hd_np = 0
                temp.append(K.constant(hd_np / max_hd))
            hds.append(K.stack(temp))

        tf_hd = K.stack(hds)

        if weights is not None:
            return K.sum(weights * tf_hd)
        return K.sum(tf_hd)
    return hd_loss_fn


def tversky_loss(alpha: float, beta: float) -> object:
    """[returns the tversky loss function]

    Args:
        alpha (float): [weighting for false positive penalty]
        beta (float): [weighting for false negative penalty]

    Returns:
        object: [tversky loss function]
    """
    #@masked
    #@generalized
    def tversky_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor,
                        weights: tf.Tensor = None) -> tf.Tensor:
        if weights is not None:
            dims = len(K.shape(y_true))
            true_pos = K.sum(weights * K.sum(y_true * y_pred, range(dims-1)))
            false_pos = alpha * \
                K.sum(weights * K.sum(y_true * (1. - y_pred), range(dims-1)))
            false_neg = beta * \
                K.sum(weights * K.sum((1. - y_true) * y_pred, range(dims-1)))
        else:
            true_pos = K.sum(y_true * y_pred)
            false_pos = alpha * K.sum(y_true * (1. - y_pred))
            false_neg = beta * K.sum((1. - y_true) * y_pred)

        tversky = (true_pos + K.epsilon()) / (true_pos + false_pos + false_neg)
        return 1 - tversky
    return tversky_loss_fn


def focal_loss(focus_param: float, balance_param: float) -> object:
    """[returns the focal loss function]

    Args:
        focus_param (float): [focusing parameter]
        balance_param (float): [balancing parameter to scale loss]

    Returns:
        object: [returns the loss function with given parameters]

    Notes:
        Compatiable with generalized and masked decorators
        Generalized decorator will make the loss function small
    """
    #@generalized
    #@masked
    def focal_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor,
                      weights: tf.Tensor = None) -> tf.Tensor:
        if weights is not None:
            y_pred = K.flatten(y_pred * weights)
            y_true = K.flatten(y_true * weights)
        else:
            y_pred = K.flatten(y_pred)
            y_true = K.flatten(y_true)
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        logit = K.log(y_pred / (1. - y_pred))
        bce = K.maximum(logit, 0) - logit*y_true + \
            K.log(1 + K.exp(-1. * K.abs(logit)))
        scaled_bce = bce / K.cast(K.prod(K.shape(y_true)), dtype=tf.float32)
        focal = K.sum((1. - y_pred)**focus_param * scaled_bce)
        return focal
    return focal_loss_fn


def sensitivity_specificity_loss(r: float) -> object:
    """[returns the sensitivity-specificity loss function]

    Args:
        r (float): [r > 0.5 biases towards sensitivity]

    Returns:
        object: [sensitivity specificity loss function]

    Notes:
        Compatiable with masked and generalized decorators
    """
    #@masked
    #@generalized
    def ss_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor,
                   weights: tf.Tensor = None) -> tf.Tensor:
        if weights is not None:
            dims = len(K.shape(y_true))
            true_pos = K.sum(weights * K.sum(y_true * y_pred, range(dims-1)))
            true_neg = K.sum(weights * K.sum((1. - y_true)
                                             * (1. - y_pred), range(dims-1)))
            false_pos = K.sum(
                weights * K.sum(y_true * (1. - y_pred), range(dims-1)))
        else:
            true_pos = K.sum(y_true * y_pred)
            true_neg = K.sum((1. - y_true) * (1. - y_pred))
            false_pos = K.sum(y_true * (1. - y_pred))
        sens = true_pos / (true_pos + false_pos)
        spec = true_neg / (true_neg + false_pos)
        return r * sens + (1. - r) * spec
    return ss_loss_fn
