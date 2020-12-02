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
from torchvision.utils import save_image
from tqdm import tqdm

from PIL import Image

##autre code##
from utils import *
from pretraitement import *
# Constantes
IMG_SIZE_L = 1226#601#1000
IMG_SIZE_H = 1000#960#1266
IMAGENET_MEAN = [0.485, 0.456, 0.406] # Moyenne pour chaque canal de couleur
IMAGENET_STD = [0.229, 0.224, 0.225] # Std pour chaque canal de couleur
STYLE_IMAGE = 'style_image'
CONTENT_IMAGE = 'content_image'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE_1 = torch.device('cuda:1')
# Variables
results = {'Name':[],
           'Shape':[],
           'Mean':[],
           'Std':[],
          }



# Télécharger la portion "features" du VGG19
# Nous n'avons pas besoin des couches de classification.

vgg =  models.vgg19(pretrained = True ,progress = False).features

# Geler les couches pré-entraînées
for param in vgg.parameters():
    param.requires_grad = False
# Si GPU disponible, monter le modèle sur le GPU
vgg.to(DEVICE)

# Étapes de prétraitement
# 1. Redimensionner l'images à la taille désirée -> (3, 256, 256)
# 2. Transformer l'image PIL en tenseur
# 3. Appliquer la normalisation ImageNet
# 4. Ajouter une dimension pour PyTorch (C,H,W) -> (B,C,H,W)
#    où B est la taille de batch (lot).
preprocessing = transforms.Compose([transforms.Resize((IMG_SIZE_L, IMG_SIZE_H)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=IMAGENET_MEAN,
                                                         std=IMAGENET_STD),
                                    AddDimension(0),
                                   ])
# TODO Q3B
# Le transforms.Compose applique séquentiellement les
# transformations.
#
# Étapes de post-traitement (voir les modules précédents)
# 1. Retirer la 1ère dimension (B,C,H,W)->(C,H,W)
# 2. Appliquer l'inverse de la normalisation ImageNet
# 3. Permuter les dimension pour Matplotlib (C,H,W)->(H,W,C)
# 4. Clamp les valeurs des tenseurs entre [0,1]
#
# Vous devez ici passer les paramètres désirés dans
# l'appel des classes Custom de transformationé.
postprocessing = transforms.Compose([RemoveDimension(0), 
                                     UnNormalize(mean=IMAGENET_MEAN,
                                                 std=IMAGENET_STD),
                                     Permute((1,2,0)),
                                     Clamp(0,1),
                                    ])

# Le code suivant est donné pour permettre d'afficher
# certaines informations utiles qui vous permettront
# de comprendre si vos transformations post-traitement
# sont fonctionnelles.


#################################################
#               Importer image             #
#################################################
# Le StyleTransfer ne requière que 2 images pour fonctionner
# 1. Une image dont vous voulez extraire le style -> Style image
# 2. Une image sur laquelle vous voulez appliquer le style -> Content image

# TODO Q3A
# Avec la fonction fetch_image, téléchargez vos propres images.
# Pour se faire, ajoutez vos images sur votre Google Drive en les
# téléversant et partagez-les avec un lien URL. Dans ce lien, 
# vous retrouverez le <FILE_ID> qu'il faut copier-coller dans la
# fonction fetch_image.
# Exemple: https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing

#dani dog
#'https://drive.google.com/file/d/1gmTXzKr2evF_M-ejgFYNrD2BikMUGYRH/view?usp=sharing'
#kath 1
#'https://drive.google.com/file/d/1czYzJkFVeMEX0_wdVSD6OJGUTOkSIrnh/view?usp=sharing'
#kath 2
#'https://drive.google.com/file/d/1K1Xr2eEj2oQzBcfSmEYb8Ib0MWmy_eiW/view?usp=sharing
#kath 3
#'https://drive.google.com/file/d/1emnh2FpTuI2yATDowN6vhvo1LV4SGLE3/view?usp=sharing'

#max
#'https://drive.google.com/file/d/1w4WcLWFVjxZps7caRdWpy9efWcLAmDZI/view?usp=sharing'
#'https://drive.google.com/file/d/1DztTzF6_jcnREVTVCeUrqxzwO2-XM57o/view?usp=drivesdk'
#vangogh portrait
#'https://drive.google.com/file/d/1aPVExH_85wQ9KtrYZY9Fy7XzXERSf38H/view?usp=sharing'

# TODO Q3A
# Télécharger une image contenant le style à extraire
style_image_file_id = "1aPVExH_85wQ9KtrYZY9Fy7XzXERSf38H"
style_image = fetch_image(style_image_file_id)

# TODO Q3A
# Télécharger une image sur laquelle appliquer le style
content_image_file_id = "1DztTzF6_jcnREVTVCeUrqxzwO2-XM57o"
content_image = fetch_image(content_image_file_id)

images = {STYLE_IMAGE:style_image,
          CONTENT_IMAGE:content_image}

# Afficher les 2 images côte-à-côte
plt.figure(figsize=(15,15))

# Affichage du style_image
plt.subplot(1, 2, 1)
plt.imshow(images[STYLE_IMAGE])

# Affichage du content_image
plt.subplot(1, 2, 2)
plt.imshow(images[CONTENT_IMAGE])

#################################################
#    Vérification des bases des paramètres      #
#################################################
# Afficher les statistiques des images naturelles
for name, img in images.items():
    results['Name'].append(f'raw_{name}')
    results['Shape'].append(img.size)
    mean = numpy.mean(img)/255
    results['Mean'].append(mean)
    std = numpy.std(img)/255
    results['Std'].append(std)

#################################################
#    Application du prétraitement des images    #
#################################################
# Appliquer le prétraitement sur les images
pre_images = {}
for k,v in images.items():
    pre_images[k] = preprocessing(v)
    pre_images[k] = pre_images[k].to(DEVICE_1)

# Afficher les statistiques des images transformées
for name, img in pre_images.items():
    results['Name'].append(f'pre_{name}')
    results['Shape'].append(img.shape)
    results['Mean'].append(img.mean().item())
    results['Std'].append(img.std().item())

#################################################
#    Test du post-traitement des images         #
#################################################
post_images = {}

# Appliquer le post-traitement sur les images
for name,img in pre_images.items():
    image = img.cpu().detach()
    post_images[name] = postprocessing(image)

# Afficher les statistiques des images transformées
for name, img in post_images.items():
    results['Name'].append(f'post_{name}')
    results['Shape'].append(img.shape)
    results['Mean'].append(img.mean().item())
    results['Std'].append(img.std().item())

# Affichage des résultats
# N.B. Bien que la taille de l'image ait été changée par le resize,
#      les valeurs de moyenne et de déviation standard devraient être
#      très proches avant le prétraitement et après le post-traitement.
df = pandas.DataFrame(results)
display.display(df)

################PREPROCESS##################
# TODO Q3E
# Extraire les features de l'image de style avec la fonction
# extract_features.
pre_style_image = preprocessing(images[STYLE_IMAGE])
pre_style_image = pre_style_image.to(DEVICE)
style_features = extract_features(pre_style_image,vgg)
# TODO Q3E
# Extraire les features de l'image de content avec la fonction
# extract_features.
pre_content_image = preprocessing(images[CONTENT_IMAGE])
pre_content_image = pre_content_image.to(DEVICE)
content_features = extract_features(pre_content_image,vgg)
# Pré-calculer la matrice de Gram pour chaque couche de style
style_grams = {}
for layer in style_features:
    style_grams[layer] = gram_matrix(style_features[layer])

# TODO Q3E
# Création d'une image cible temporaire.
# Utilisez la fonction clone() de la
# la librairie PyTorch. N'oubliez pas
# le gradient! Considérez également le
# device utilisé (CPU vs GPU).
# Il faut travailler sur une copie de
# l'image cible pour changer son style
# itérativement.
target = torch.clone(pre_content_image)
target.requires_grad_(True)
target.to(DEVICE)
# Poids appliqués pour chaque couche de style 
# Valeurs par défaut:
# 'conv1_1': 1.
# 'conv2_1': 0.75
# 'conv3_1': 0.2
# 'conv4_1': 0.2
# 'conv5_1': 0.2
style_layers_weights = {'conv1_1': 1.,
                        'conv2_1': 0.75,
                        'conv3_1': 0.2,
                        'conv4_1': 0.5,
                        'conv5_1': 1}

# Par défaut: content_weight = 1
content_weight = 1 

# Par défaut: style_weight = 1e7
style_weight = 1e12


# TODO Q3F
# Implémenter la fonction qui calcule la loss de contenu pour
# une couche donnée.
# La loss de contenu est calculée par la MSE entre les
# paramètres de l'image cible et les paramètres de l'image
# de contenu pour chaque couche.
def calculate_content_loss(layer_name):
    assert(layer_name in target_features.keys())
    assert(layer_name in content_features.keys())
    """
    Calculates the content loss between the target image features and
    the content image features.

    Args:
        layer_name (String) : Name of the layer to evaluate.

    Returns:
        tensor (Tensor): Tensor containing the loss of the squared mean
                         difference between the target and content layers.
    """
    content_loss = torch.sum(torch.square(target_features[layer_name] - content_features[layer_name]))
    return content_loss

# TODO Q3F
# Implémenter la fonction qui calcule la loss de style pour
# une couche donnée.
# La loss de style est calculée par la MSE entre la matrice Gram
# de contenu et la matrice Gram de style, pondérée par le poids
# donné à chaque couche.
def calculate_style_loss(weight_layer, target_gram, style_gram, target_feature):
    """
    Calculates the style loss between the Gram matrix of the image features and
    the Gram matrix of the content image features.

    Args:
        weight_layer (Float) : weighting for the current layer (w_l).
        target_gram (Tensor) : Gram matrix of the target image (G).
        style_gram (Tensor) : Gram matrix of the style image (A).
        target_feature (Tensor) : tensor containing the target feature for
                                  the current layer.

    Returns:
        style_loss (Float): computed style loss for the current layer. 
    """
    a, N, h, w = target_features[layer_name].size()
    M = h * w
    style_loss = weight_layer * torch.sum(torch.square(target_gram - style_gram))
    style_loss = style_loss  / (4* M**2 * N**2)
    return style_loss

# TODO Q3F
# Implémenter la fonction qui calcule la loss totale pour l'itération.
# La loss totale est calculée par la somme pondérée de la loss de contenu
# et la loss de style.
def calculate_total_loss(content_weight, content_loss, style_weight, style_loss):
    """
    Calculates the total loss for the current iteration.

    Args:
        content_weight (Float) : Alpha weighting for the content.
        content_loss (Float) : Content loss.
        style_weight (Float) : Beta weighting for the style.
        style_loss (Float) : Total loss.

    Returns:
        total_loss (Float): computed total loss for the current iteration. 
    """
    total_loss = content_weight * content_loss + style_weight * style_loss
    return total_loss

# Nombre total d'itérations pour appliquer
# le transfert de style.
# (Min: 2000 | Recommandé: 5000)
steps = 5000

# Fréquence de mise à jour de l'image
# (Valeur recommandée: 500)
show_image_every = 500

# Initialisation de l'optimizer Adam
# Puisqu'on modifie la cible, on l'applique
# directement sur les pixels de l'image
# (learning_rate = 3e-3)
#
# [Gatys et al, 2016] fait usage de L-BFGS mais
# pour simplifier l'implémentation et accélérer
# la convergence vers des résultats visibles, 
# Adam est plus approprié.
optimizer = optim.Adam([target], lr=3e-3)

for s in tqdm(range(1, steps+1)):

    # 1. Remettre les gradients à zéro
    optimizer.zero_grad()

    # 2. Extraire les features de l'image cible
    target_features = extract_features(target, vgg)

    # TODO Q3F
    # 3. Calculer la loss de content avec la fonction
    # calculate_content_loss(). Vous devez retrouver le
    # nom de la couche dans le graphe du modèle.
    layer_name = "conv4_1"
    content_loss = calculate_content_loss(layer_name)

    #4. Calculer la loss de style en accumulant sa valeur pour chaque couche
    style_loss = 0 # Initialiser la loss de style à zéro
    for l_name, l_weight in style_layers_weights.items():

        # Extraire le content pour la couche
        target_feature = target_features[l_name]

        # Calculer matrice Gram du content
        target_gram = gram_matrix(target_feature)

        # Extraire la matrice Gram pré-calculée pour le style
        style_gram = style_grams[l_name]

        # TODO Q3F
        # Calculer la loss de style avec pondération pour 
        # la couche donnée avec la fonction
        # calculate_style_loss().
        layer_style_loss = calculate_style_loss(l_weight,
                                                target_gram,
                                                style_gram,
                                                target_feature)

        # Accumuler la loss de style
        style_loss += layer_style_loss

    # TODO Q3F
    #5. Calculer la loss totale avec la fonction
    # calculate_total_loss()
    total_loss = calculate_total_loss(content_weight,
                                  content_loss,
                                  style_weight,
                                  style_loss)

    #6. Mettre à jour l'image cible
    total_loss.backward()
    optimizer.step()

    # Afficher les images intermédiaires
    if  s % show_image_every == 0:
        # Appliquer le postprocessing sur les images
        plt.figure(figsize=(10,10))
        img = target.cpu().detach()
        img_post = postprocessing(img)
        plt.imshow(img_post)
        save_image(img,'output/img'+ str(s) + '.png')
        plt.axis('off')
        plt.show()

# Libère la cache sur le GPU
# *Important sur un cluster de GPU*

