import numpy as np

def preprocess_images(images):
    images = images.astype(np.float32) / 255.0
    images = images.reshape(-1, 28*28)
    return images

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]
