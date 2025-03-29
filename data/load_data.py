from tensorflow.keras.datasets import mnist

def load_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    return train_images, train_labels, test_images, test_labels
