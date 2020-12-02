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
        plt.axis('off')
        plt.show()

# Libère la cache sur le GPU
# *Important sur un cluster de GPU*


