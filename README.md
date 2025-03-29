# Neural Network from Scratch - MNIST Digit Recognition

## This project implements a neural network from scratch (without using machine learning libraries like TensorFlow or PyTorch) to recognize handwritten digits from the MNIST dataset.

Read the full article on Medium

The dataset is loaded using:
from tensorflow.keras.datasets import mnist

## Project Structure

├── src/                # Source code for the neural network
│   ├── neuron.py       # Neuron class
│   ├── layer.py        # Layer class
│   ├── network.py      # Network architecture and training
│   ├── train.py        # Training loop
│   └── test.py         # Evaluation code
├── data/               # (Optional) Data loading or storage
├── notebooks/          # Jupyter notebooks (for experiments or visualizations)
├── models/             # Saved model weights (optional)
├── README.md           # This file
├── requirements.txt    # Python dependencies
└── .gitignore          # Files ignored by Git


## Example Results

| Epoch | Learning Rate | Layers               | Accuracy |
|-------|---------------|----------------------|----------|
| 1     | 0.001         | [256, 128, 64]       | 82.06%   |
| 1     | 0.001         | [350]                | 93.30%   |
| 1     | 0.001         | [256, 128, 64, 32]   | 75.48%   |
| 2     | 0.001         | [350]                | 94.66%   |
| 10    | 0.001         | [350]                | 87.77%   |
| 1     | 0.0001        | [350]                | 76.05%   |
| 3     | 0.001         | [350]                | 95.55%   |
| 4     | 0.001         | [350]                | 77.20%   |


## Technologies Used

Python 3.11

NumPy

MNIST Dataset (via tensorflow.keras.datasets)

## Core Concepts

- Forward Propagation

- Backpropagation

- Gradient Descent

- Weight & Bias Initialization

- Cost Function (Mean Squared Error)

## License

MIT License © 2025 Ammar Souchon

## Author

AmmarSo Follow me on Medium 

## Show your support

If you like this project, please consider leaving a ⭐ on the repository!
