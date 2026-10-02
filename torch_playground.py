import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ---- device setup ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

torch.set_default_device(device)
torch.manual_seed(0)

print(f"PyTorch {torch.__version__} | device: {device}")
# ---- code ----

"Settings"
N = 10                         # batch size
D_in = 1                       # inputs/batch
D_out = 1                      # outputs/batch

learning_rate = 0.1
epochs = 1000

X = torch.randn(N, D_in)       # random tensor
W_true = torch.tensor([[2.0]]) # our set weight
b_true = torch.tensor([1.0])   # our set bias
y_true = X @ W_true + b_true   # expected value of y

"1 manual epoch"
# W = torch.rand(D_in, D_out, requires_grad=True) # random weight
# b = torch.rand(D_out, requires_grad=True)       # random bias

# print(f"Initial W: {W}\n")
# print(f"Initial b: {b}\n")

# y_hat = X @ W + b # predicted value of y

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

"1000 automated epochs"
# W_true = torch.tensor([[2.0]])
# b_true = torch.tensor([1.0])
# X = torch.rand(N, D_in)
# y = X @ W_true + b_true

# W = torch.rand(D_in, D_out, requires_grad=True)
# b = torch.rand(D_out, requires_grad=True)

# for epoch in range(epochs):
#     y_hat = X @ W + b

#     loss = torch.mean((y_hat - y) ** 2)

#     loss.backward()

#     with torch.no_grad():
#         W -= learning_rate * W.grad
#         b -= learning_rate * b.grad

#     W.grad.zero_()
#     b.grad.zero_()

#     if epoch % 100 == 0:
#         print(f"Epoch: {epoch:02d}; y_hat: {y_hat}; Loss {loss.item():.4f}; W: {W.item():.3f}; b: {b.item():.3f}\n")

# print(f"Final W:{W.item():.3f}\n")
# print(f"Final b:{b.item():.3f}\n")

"torch.nn framework speedrun"
"1. Linear layer"
# linear_layer = nn.Linear(in_features=D_in, out_features=D_out)

# print(f"W: {linear_layer.weight}\n")
# print(f"b: {linear_layer.bias}\n")

# y_hat_nn = linear_layer(X)

# print(f"Output of nn_linear: {y_hat_nn}\n")

"2. Embedding"
# vocab_size = 70   # our dict: 10 unique words
# embedding_dim = 3 # each word = 3D vector

# embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# input_ids = torch.tensor([[1, 36, 67, 69]])
# word_vectors = embedding_layer(input_ids)

# print(embedding_layer)
# print(word_vectors)

"3. Dropout (prevent overfitting)"
# dropout_layer = nn.Dropout(0.5) # turns dropout on
# input_tensor = torch.ones(1,10)

# # training
# dropout_layer.train() # turns dropout on 
# output_during_train = dropout_layer(input_tensor)
# print(f"Outputs during training (randomly zeroed): {output_during_train}")

# # evaluation/prediction
# dropout_layer.eval() # turns dropout off 
# output_during_eval = dropout_layer(input_tensor)
# print(f"Output during evaluation: {output_during_eval}")

"4. Module"
class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_layer = nn.Linear(in_features, out_features)

    def forward(self, X):
        return self.linear_layer(X)

model = LinearRegression(in_features=1, out_features=1)
print(model)

"5. Optimizer"
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # model.parameters() -> which tensors to manage
loss_fn = nn.MSELoss()

"6. 3 Golden Lines"
# # 1.
# optimizer.zero_grad()
# # 2.
# loss.backward()
# # 3.
# optimizer.step()

"7. Full loop"
for epoch in range (epochs) : 
    y_hat = model(X)
    loss = loss_fn(y_hat, y_true)
    # loss = F.mse_loss(y_hat, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0 : 
        print(f"Epoch: {epoch}; Loss: {loss.item():.3f}")
