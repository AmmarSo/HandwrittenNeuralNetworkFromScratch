import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Charger le dataset MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Choisir un index pour récupérer une image (par exemple, la première image)
index = 0
image = train_images[index]
label = train_labels[index]

# Afficher l'image
plt.imshow(image, cmap='gray')
plt.title(f'Label: {label}')
plt.show()

# Sauvegarder l'image en local si nécessaire
plt.imsave('mnist_digit.png', image, cmap='gray')
