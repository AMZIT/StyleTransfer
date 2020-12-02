import numpy
import os
import requests
import time
import pandas
pandas.set_option('display.max_colwidth', 0)

from IPython import display
import matplotlib.pyplot as plt
from io import BytesIO
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm

from PIL import Image

#################################################
#          Prétraitement des images             #
#################################################

# Classe de transformation Custom
# pour ajouter un channel à la position "dim".
# Cette classe vous est donnée comme exemple
# pour l'implémentation des autres transformations.
class AddDimension(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        """
        Args:
            tensor (Tensor): Tensor image of size (C,H,W).

        Returns:
            tensor (Tensor): Tensor image with an added channel, now of size (1,C,H,W).
        """

        new_x = x.unsqueeze(self.dim)
        return new_x


#################################################
#         Post-traitement des images            #
#################################################

# Puisque PyTorch travaille sur des lots (batch)
# d'images, une dimension supplémentaire (B) est
# ajoutée pour son fonctionnement. L'image en entrée
# passe donc de la taille (3, 256, 256) à la taille
# (B, 3, 256, 256) ou B, ici, est de taille 1, car il
# n'y a qu'une seule image par lot.
#
# Toutefois, afin d'afficher l'image hybride, il est important
# de retirer cette dimension supplémentaire, car les outils
# d'affichage s'attendent à afficher une image unique, 
# Cette prochaine classe doit donc vous permettre de 
# retirer cette dimension supplémentaire. 
class RemoveDimension(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        """
        Args:
            x (Tensor): Tensor image of size (1,C,H,W).

        Returns:
            new_x (Tensor): Tensor image with the removed channel, now of size (C,H,W).
        """

        # TODO Q3B
        # Implémentation d'une transformation Custom
        # pour retirer un channel à la position "dim"
        # Utilisez la fonction Pytorch: Squeeze()
        new_x = x.squeeze(self.dim)
        return new_x

# Comme le réseau VGG19 a été pré-entraîné sur ImageNet
# avec des paramètres de normalisation spécifique à ce
# jeu de données, on assume qu'il sera plus performant sur
# une nouvelle distribution d'images si celle-ci partage
# également cette normalisation. Ainsi, pour l'extraction
# des features de style et de contenu, le modèle VGG19 doit
# travailler sur des images normalisées.
#
# Toutefois, afin d'afficher l'image hybride, il est
# important de retirer la normalisation si on 
# souhaite avoir un résultat visuellement intéressant,
# car la normalisation a un impact sur la distribution
# des valeurs de pixels dans l'image. La classe suivante
# doit vous permettre d'appliquer l'inverse de la
# normalisation.
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        """
        Args:
            x (Tensor): Tensor image of shape (C,H,W).
        Returns:
            new_x (Tensor): UnNormalized tensor image (C,H,W).
        """

        # TODO Q3B
        # Implémentation d'une transformation Custom
        # pour appliquer l'inverse de la normalisation.
        # Vous devez implémenter cette fonction manuellement
        # en utilisant des opérations sur les tenseurs.
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        new_x = x.mul_(std.view(-1, 1, 1)).add_(mean.view(-1, 1, 1))
        return new_x


# Pour que les images s'affichent, la valeur des pixels doit
# être retournée entre [0,1] (ou [0,255], mais veuillez utiliser
# ici la plage [0,1]). En effet, les fonctions d'activation
# du réseau de neurone actuel ne bornent pas les valeurs des 
# pixels dans la plage [0,1]. Il faut alors alors ramener les
# valeurs des pixels qui sortent de ces bornes à l'intérieur de 
# ces dernières.
# 
# La classe suivante dois vous permettre de borner les valeurs
# des pixels de l'image hybride entre [0,1].
class Clamp(object):
    def __init__(self, min, max):
        self.min = float(min)
        self.max = float(max)

    def __call__(self, x):
        """
        Args:
            x (Tensor): Tensor of the image.

        Returns:
            new_x (Tensor): Tensor with values clipped within [0,1].
        """

        # TODO Q3B
        # Implémentation d'une transformation Custom
        # pour borner les valeurs dans la plage [0, 1]
        # Utilisez la fonction PyTorch: Clamp()
        new_x = torch.clamp(x,self.min,self.max)
        return new_x

# Pour que la librairie Matplotlib puisse afficher le
# contenu des images, il est important que les canaux
# soient donnés dans le bon ordre.
#
# PyTorch utilise les images sous la forme (B,C,H,W)
# et Matplotlib doit recevoir les images sous la forme
# (H,W,C). Comme le Permute est appelé après le
# RemoveDimension(), vous aurez ici, en entrée, un tenseur
# (C,H,W) que vous devez transformer dans la forme
# désirée pour l'affichage de Matplotlib.
class Permute(object):
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, x):
        """
        Args:
            x (Tensor): Tensor of the image.

        Returns:
            new_x (Tensor): Tensor of the image with permuted dimensions.
        """

        # TODO Q3B
        # Implémentation d'une transformation Custom pour
        # faire un Permute, car Matplotlib lit les
        # images en format (H,W,C).
        # Utilisez la fonction PyTorch: Permute()
        new_x = x.permute(self.dims[0],self.dims[1],self.dims[2])
        return new_x


