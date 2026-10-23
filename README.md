# MNIST Neural Network from Scratch

A handwritten digit classifier built with **NumPy**, with explicit implementations of **forward propagation, backpropagation, and gradient-based parameter updates**. The project makes the mechanics of neural-network training visible, from individual matrix operations to digit predictions.

The network learns from **60,000 MNIST images** using a **784 → 128 → 10** architecture. The model achieves **97.43% test accuracy**. Average training cross-entropy decreases from **0.4651 to 0.0542 over 10 epochs**.

## Project highlights

- **Neural-network fundamentals:** Implement weight initialization, activation functions, cross-entropy loss, gradients, and parameter updates directly in NumPy.
- **Multiclass classification:** Recognize handwritten digits from **0 to 9** using a softmax output layer.
- **Data preparation:** Normalize pixel intensities and one-hot encode training labels.
- **Visual exploration:** Display a labeled 3 × 3 grid of sample digits with Matplotlib.
- **Training and inference:** Track average loss across epochs and generate predictions for test images.

The model and training loop use NumPy; pandas handles CSV data, and Matplotlib handles visualization.

## Network architecture

| Layer | Size | Activation | Role |
| --- | ---: | --- | --- |
| Input | 784 | — | Flattened 28 × 28 grayscale image |
| Hidden | 128 | Sigmoid | Learned nonlinear representation |
| Output | 10 | Softmax | Probabilities for digits 0–9 |

The network contains **101,770 trainable parameters**, including weights and biases.

### Training configuration

| Setting | Value |
| --- | --- |
| Training examples | 60,000 |
| Epochs | 10 |
| Learning rate | 0.01 |
| Update frequency | After each training example |
| Loss | Multiclass cross-entropy |
| Weight initialization | Random normal values scaled by 0.01 |
| Bias initialization | Zeros |
| Pixel normalization | Divide by 255 |

## Results

Average training cross-entropy decreases throughout the ten epochs:

| Epoch | Average training loss |
| ---: | ---: |
| 1 | 0.4651 |
| 2 | 0.2143 |
| 3 | 0.1593 |
| 4 | 0.1266 |
| 5 | 0.1050 |
| 6 | 0.0895 |
| 7 | 0.0777 |
| 8 | 0.0683 |
| 9 | 0.0606 |
| 10 | **0.0542** |

These values describe the model's training loss. The prediction stage assigns each test image the digit with the highest output probability.

### Test accuracy

**Test accuracy: 97.43%.**

The 17th notebook cell compares the predicted digits with the ground-truth labels in `mnist_test.csv` and prints **test accuracy as a percentage**:

```python
test_accuracy = np.mean(np.array(predictions) == test_data.iloc[:, 0].values)
```

Run the this cell after training and prediction to reproduce the evaluation. Results may vary between runs because the weights are randomly initialized without a fixed seed.

## How it works

1. **Load and explore:** Read the training CSV, inspect its structure, and visualize sample digits.
2. **Prepare inputs:** Scale the 784 pixel values to the range `[0, 1]` and convert labels to one-hot vectors.
3. **Initialize parameters:** Create small random weights and zero biases for both dense layers.
4. **Forward pass:** Calculate hidden activations with sigmoid, then class probabilities with softmax.
5. **Calculate loss:** Measure the difference between the predicted probabilities and the target class using cross-entropy.
6. **Backpropagate:** Compute gradients for the output layer and propagate them through the sigmoid hidden layer.
7. **Update parameters:** Apply gradient-descent updates after each example and repeat for ten epochs.
8. **Predict:** Apply the trained network to test images and select the most probable digit with `argmax`.

## Dataset

The project uses **MNIST**, a dataset of 28 × 28 grayscale handwritten digits. A compatible CSV download is available from [MNIST in CSV on Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

Download and extract these files into the same directory as the notebook:

| File | Examples | Expected layout |
| --- | ---: | --- |
| `mnist_train.csv` | 60,000 | Header row, then a `label` column followed by 784 pixel columns |
| `mnist_test.csv` | 10,000 | Header row, then a label column followed by 784 pixel columns |

Both files must include the leading label column: the notebook selects pixel features from the second column onward. Use the labeled CSV files linked above for this layout.

MNIST was developed by **Yann LeCun, Corinna Cortes, and Christopher J. C. Burges**. See the [MNIST dataset mirror maintained by the CVDF](https://github.com/cvdfoundation/mnist) for the original-format data and dataset background.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/K1M1A-K1/MNIST-Neural-Network-From-Scratch.git
cd MNIST-Neural-Network-From-Scratch
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Dependencies: **NumPy**, **pandas**, **Matplotlib**, and **Jupyter**.

### 3. Add the dataset

Place `mnist_train.csv` and `mnist_test.csv` alongside `Neural_Network_From_Scratch.ipynb` using the filenames shown above.

### 4. Open the notebook

```bash
jupyter notebook Neural_Network_From_Scratch.ipynb
```

Run all cells in order to explore the data, train the model, generate predictions, and print test accuracy in the final cell. Training processes one image at a time, so runtime depends on your CPU and NumPy configuration. No GPU is required.

## Project files

| File | Purpose |
| --- | --- |
| `Neural_Network_From_Scratch.ipynb` | Data exploration, neural-network implementation, training, and prediction |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview, architecture, training results, and setup |
| `mnist_train.csv` | Training data downloaded separately |
| `mnist_test.csv` | Test data downloaded separately |

## References

The notebook credits the following implementation and learning resources:

- [Om Baval — Building a Simple Neural Network from Scratch for MNIST Digit Recognition](https://medium.com/@ombaval/building-a-simple-neural-network-from-scratch-for-mnist-digit-recognition-without-using-7005a7733418).
- [Om Baval — Neural Network from Scratch implementation](https://github.com/OmBaval/Neural-Network-from-scratch-without-TensorFlow-PyTorch/blob/main/Neural%20Network%20from%20Scratch.ipynb).
- [NumPy — The Absolute Basics for Beginners](https://numpy.org/doc/stable/user/absolute_beginners.html).
- [Activation Functions: Sigmoid, Tanh, ReLU, Leaky ReLU, and Softmax](https://medium.com/@cmukesh8688/activation-functions-sigmoid-tanh-relu-leaky-relu-softmax-50d3778dcea5).

Additional resources are listed in the notebook's References section.
