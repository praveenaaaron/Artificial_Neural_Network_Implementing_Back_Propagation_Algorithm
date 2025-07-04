
## 🔢 Training Data

```python
X = np.array([[2, 9], [1, 5], [3, 6]], dtype=float)
y = np.array([[92], [86], [89]], dtype=float)
```

### 📊 What this means:

This is your dataset with **3 training examples**.

| Input 1 (Hours Studied) | Input 2 (Hours Slept) | Output (Marks) |
| ----------------------- | --------------------- | -------------- |
| 2                       | 9                     | 92             |
| 1                       | 5                     | 86             |
| 3                       | 6                     | 89             |

---

## 🔄 Data Normalization

```python
X = X / np.amax(X, axis=0)  # Normalize features
y = y / 100                 # Normalize target
```

### ✅ Why:

* Neural networks learn better when data is scaled to a small range (like 0 to 1).
* This avoids very large gradients or slow learning.

After normalization:

```python
X = [[0.666, 1.0], [0.333, 0.555], [1.0, 0.666]]
y = [[0.92], [0.86], [0.89]]
```

---

## 🧠 Neural Network Structure

```python
inputlayer_neurons = 2
hiddenlayer_neurons = 3
output_neurons = 1
```

### 🔧 Configuration:

* 2 inputs → Hours studied & slept
* 3 hidden layer neurons
* 1 output → Marks predicted

---

## 🏁 Initialization

```python
wh = np.random.uniform(size=(2, 3))    # Weights from input to hidden
bh = np.random.uniform(size=(1, 3))    # Bias for hidden layer
wout = np.random.uniform(size=(3, 1))  # Weights from hidden to output
bout = np.random.uniform(size=(1, 1))  # Bias for output layer
```

> These are **random numbers**, meaning the model starts with random guesses and learns through training.

---

## 🔁 Training Loop

```python
for i in range(epoch):  # Train for 5000 times
```

Each iteration (epoch) includes:

---

### 1. **Forward Propagation**

```python
hinp1 = np.dot(X, wh)
hinp = hinp1 + bh
hlayer_act = sigmoid(hinp)
```

* Computes input to hidden layer.
* Applies sigmoid activation to get `hlayer_act`.

```python
outinp1 = np.dot(hlayer_act, wout)
outinp = outinp1 + bout
output = sigmoid(outinp)
```

* Computes input to output layer.
* Applies sigmoid to get **final predicted output**.

---

### 2. **Backpropagation**

```python
EO = y - output               # Error at output
outgrad = derivatives_sigmoid(output)
d_output = EO * outgrad       # Gradient at output
```

* Calculates the error and gradient for output layer.

```python
EH = d_output.dot(wout.T)     # Error propagated back to hidden layer
hiddengrad = derivatives_sigmoid(hlayer_act)
d_hiddenlayer = EH * hiddengrad
```

* Calculates how much hidden layer contributed to the error.

---

### 3. **Update Weights**

```python
wout += hlayer_act.T.dot(d_output) * lr
wh += X.T.dot(d_hiddenlayer) * lr
```

* Adjusts weights based on error gradients and learning rate `lr = 0.1`.

```python
bout += np.sum(d_output, axis=0, keepdims=True) * lr
bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr
```

* Adjusts biases similarly.

---

## 📤 After Training

```python
print("Input:\n" + str(X))
print("Actual Output:\n" + str(y))
print("Predicted Output:\n", output)
```

### Output might look like:

```text
Input:
[[0.666 1.0]
 [0.333 0.555]
 [1.0   0.666]]
Actual Output:
[[0.92]
 [0.86]
 [0.89]]
Predicted Output:
[[0.91]
 [0.85]
 [0.89]]
```

The neural network has **learned** to predict the outputs close to the actual values after training!

---

## 📌 Summary:

| Step                   | Description                                      |
| ---------------------- | ------------------------------------------------ |
| **1. Input**           | Feed inputs (hours of study/sleep).              |
| **2. Normalize**       | Scale to 0–1.                                    |
| **3. Forward Pass**    | Compute predictions.                             |
| **4. Backpropagation** | Compute error and update weights.                |
| **5. Output**          | After 5000 iterations, get accurate predictions. |


