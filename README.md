🧠 Neural Network from Scratch - MNIST Digit Recognition

This project implements a neural network from scratch (without using machine learning libraries like TensorFlow or PyTorch) to recognize handwritten digits from the MNIST dataset.

📘 Read the full article on Medium

📂 Project Structure

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

🚀 Getting Started

1. Clone the repository

git clone https://github.com/<your-username>/mnist-neural-network-from-scratch.git
cd mnist-neural-network-from-scratch

2. Create a virtual environment (optional)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Run the training

python src/train.py

5. Evaluate the model

python src/test.py

📊 Example Results

Metric

Value

Accuracy

~92%

Loss

Decreasing

Hidden Layers

2

Activation

Sigmoid

🧪 Technologies Used

Python 3.x

NumPy

MNIST Dataset (from Yann LeCun's website)

🧠 Core Concepts

Forward Propagation

Backpropagation

Gradient Descent

Weight & Bias Initialization

Cost Function (Mean Squared Error)

📄 License

MIT License © 2025 Ammar Souchon

✍️ Author

Ammar SouchonFollow me on Medium

🌟 Show your support

If you like this project, please consider leaving a ⭐ on the repository!

