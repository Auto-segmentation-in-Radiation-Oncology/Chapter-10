import tensorflow.keras.backend as K
import torch
import numpy as np


# Numpy implementations
class numpy_losses:
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta
        self._eps = np.finfo(float).eps

    def dice_coef(self, output, labels):
        # Computes the dice coefficient of two numpy arrays
        intersection = np.sum(output * labels)
        denominator = np.sum(output) + np.sum(labels)
        return (2 * intersection + self._eps) / (denominator + self._eps)

    def dice_loss(self, output, labels):
        # Computes the dice loss of two numpy arrays
        return 1 - self.dice_coef(output, labels)

    def hinge(self, output, labels):
        hinge_loss = 1 - (output * labels)
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss

    def generalized_dice_coef(self, output, labels):
        # Computes the generalized dice coefficient of two numpy arrays
        sum_dims = tuple(range(labels.ndim))
        w = 1 / (np.sum(labels, axis=sum_dims[:-1])**2 + self._eps)
        numerator = np.sum(w * np.sum(output * labels, axis=sum_dims))
        denominator = np.sum(w * np.sum(output + labels, axis=sum_dims))
        return (2 * numerator + self._eps) / (denominator + self._eps)

    def generalized_dice_loss(self, output, labels):
        # Computes the generalized dice loss of two numpy arrays
        return 1 - self.generalized_dice_coef(output, labels)

    def tversky_coef(self, output, labels):
        # Calculates the tversky coefficient or loss
        t_pos = np.sum(output * labels)
        f_pos = self.alpha * np.sum(labels * (1 - output))
        f_neg = self.beta * np.sum(output * (1 - labels))
        return (t_pos + self._eps) / (t_pos + f_pos + f_neg + self._eps)

    def tversky_loss(self, output, labels):
        return 1 - self.tversky(output, labels)


# Keras implementations
class keras_losses:
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta
        self._eps = K.epsilon()

    def dice_coef(self, y_true, y_pred):
        # Computes the dice coefficient of two Keras tensors
        intersection = K.sum(y_true * y_pred)
        denominator = K.sum(y_true) + K.sum(y_pred)
        return (2. * intersection + self._eps) / (denominator + self._eps)

    def dice_loss(self, y_true, y_pred):
        # Computes the dice loss of two Keras tensors
        return 1 - self.dice_coef(y_true, y_pred)

    def generalized_dice_coef(self, y_true, y_pred):
        # Computes the generalized dice coefficient of two Keras tensors
        sum_dims = tuple(range(K.ndim(y_pred)))
        w = 1 / (K.sum(y_true, axis=sum_dims[:-1])**2 + self._eps)
        numerator = K.sum(w * K.sum(y_true * y_pred, axis=sum_dims))
        denominator = K.sum(w * K.sum(y_true + y_pred, axis=sum_dims))
        return (2 * numerator + self._eps) / (denominator + self._eps)

    def generalized_dice_loss(self, y_true, y_pred):
        # Computes the generalized dice loss of two Keras tensors
        return 1 - self.generalized_dice_coef(y_true, y_pred)

    def tversky_coef(self, y_true, y_pred):
        # Calculates the tversky coefficient or loss
        t_pos = K.sum(y_true * y_pred)
        f_pos = self.alpha * K.sum(y_pred * (1 - y_true))
        f_neg = self.beta * K.sum(y_true * (1 - y_pred))
        return (t_pos + self._eps) / (t_pos + f_pos + f_neg + self._eps)

    def tversky_loss(self, y_true, y_pred):
        return 1 - self.tversky_coef(y_true, y_pred)


# PyTorch implementation
class torch_losses(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        #super(torch_losses, self).__init()
        self.alpha = alpha
        self.beta = beta
        self._eps = 1e-7

    def hinge(self, output, labels):
        hinge_loss = 1 - torch.mul(output, labels)
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss

    def dice_coef(self, output, labels):
        # Computes the dice coefficient of two PyTorch tensors
        output_f = output.view(-1)
        labels_f = labels.view(-1)
        intersection = torch.sum(output_f * labels_f, dim=-1)
        denominator = torch.sum(output_f, dim=-1) + torch.sum(labels_f, dim=-1)
        return (2 * intersection + self._eps) / (denominator + self._eps)

    def dice_loss(self, output, labels):
        return 1 - self.dice_coef(output, labels)

    def no_bkgd_dice_loss(self, output, labels):
        # Computes the dice loss of two PyTorch tensors
        return 1 - self.dice_coef(output[..., 1:], labels[..., 1:])

    def generalized_dice_coef(self, output, labels):
        # Computes the generalized dice coefficient of two PyTorch tensors
        sum_dims = tuple(range(labels.dim()))
        w = 1 / (torch.sum(labels, dim=sum_dims[:-1])**2 + self._eps)
        numerator = torch.sum(w * torch.sum(output * labels, dim=sum_dims))
        denominator = torch.sum(w * torch.sum(output + labels, dim=sum_dims))
        return (2 * numerator + self._eps) / (denominator + self._eps)

    def generalized_dice_loss(self, output, labels):
        # Computes the generalized dice loss of two PyTorch tensors
        return 1 - self.generalized_dice_loss(output, labels)

    def tversky_coef(self, output, labels):
        # Calculates the tversky coefficient or loss
        t_pos = torch.sum(output * labels)
        f_pos = self.alpha * torch.sum(labels * (1 - output))
        f_neg = self.beta * torch.sum(output * (1 - labels))
        return (t_pos + self._eps) / (t_pos + f_pos + f_neg + self._eps)

    def tversky_loss(self, output, labels):
        return 1 - self.tversky_coef(output, labels)
