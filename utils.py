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

def fetch_image(file_id):
    """
    Cette fonction télécharge une image que vous partagez de votre Google Drive.
    Elle retourne l'image dans un format PIL.
    """
    URL = "https://drive.google.com/uc?"
    session = requests.Session()
    
    r = session.get(URL, params = { 'id' : file_id, 'alt' : 'media'}, stream = True)
    error_msg = f'ERROR: impossible to download the image (code={r.status_code})'
    assert(r.status_code == 200), error_msg
    
    params = { 'id' : file_id, 'confirm' : 'download_warning' }
    r = session.get(URL, params = params, stream = True)
    stream = BytesIO(r.content)
    image = Image.open(stream)
    return image
    
# Gram matrix
def gram_matrix(tensor):
    """ Calcul de la matrice de Gram pour un tenseur donné 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    # Get the (B, C, H, W) of the Tensor
    _, d, h, w = tensor.size()
    
    # Reshape tensor to multiply the features
    # for each channel
    tensor = tensor.view(d, h * w)
    
    # Calculate the Gram matrix
    gram = torch.mm(tensor, tensor.t())
    
    return gram

def extract_features(image, model, layers=None):
    """Infère l'image dans le modèle et extrait les features pour
       les couches désirées. Les couches par défaut concordent
       avec celles du réseau VGG19 de Gatys et al. (2016).
    """

    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features