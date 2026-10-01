import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ---- device setup ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

torch.set_default_device(device)  # new tensors land on the GPU automatically
torch.manual_seed(0)

print(f"PyTorch {torch.__version__} | device: {device}")

# ---- write your code below ----

# N = 10      # batch size
# D_in = 1    # #_inputs/batch
# D_out = 1   # #_outputs/batch

# W_true = torch.tensor([[2.0]]) # our set weight
# b_true = torch.tensor([1.0])   # our set bias
# X = torch.randn(N, D_in)       # random tensor

# y_true = X @ W_true + b_true   # expected value of y

# W = torch.rand(D_in, D_out, requires_grad=True) # random weight
# b = torch.rand(D_out, requires_grad=True)       # random bias

# print(f"Initial W: {W}\n")
# print(f"Initial b: {b}\n")

# y_hat = X @ W + b           # predicted value of y

# print(f"Prediction: {y_hat}\n")
# print(f"Actual: {y_true}\n")

# # loss calculation
# error = y_hat - y_true
# sqr_error = error ** 2
# loss = sqr_error.mean()

# print(f"Loss: {loss}\n")

# loss.backward() # back prop

# print(f"Gradient for W: \n{W.grad}\n")
# print(f"Gradient for b: \n{b.grad}\n")

learning_rate, epochs = 0.1, 1000

N, D_in, D_out = 10, 1, 1
W_true = torch.tensor([[2.0]])
b_true = torch.tensor([1.0])
X = torch.rand(N, D_in)
y = X @ W_true + b_true

W = torch.rand(D_in, D_out, requires_grad=True)
b = torch.rand(D_out, requires_grad=True)

for epoch in range(epochs):
    y_hat = X @ W + b

    loss = torch.mean((y_hat - y) ** 2)

    loss.backward()

    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad

    W.grad.zero_()
    b.grad.zero_()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch:02d}; Loss {loss.item():.4f}; W: {W.item():.3f}; b: {b.item():.3f}\n")

print(f"Final W:{W.item():.3f}\n")
print(f"Final b:{b.item():.3f}\n")