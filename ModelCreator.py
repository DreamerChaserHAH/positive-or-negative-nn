import numpy as np
import torch
import torch.optim as optim
from torch import nn
import pandas as pd
import torch.nn.functional as F
import math

def multi_step(value):
    if value < 0.1:
        return 0
    if value < 0.2:
        return 0.1
    if value < 0.3:
        return 0.2
    if value < 0.4:
        return 0.3
    if value < 0.5:
        return 0.4
    if value < 0.6:
        return 0.5
    if value < 0.7:
        return 0.6
    if value < 0.8:
        return 0.7
    if value < 0.9:
        return 0.8
    if value < 1.0:
        return 0.9
    return 1.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

training_dataset_input = pd.read_csv("dataset_input.csv")
training_dataset_output = pd.read_csv("dataset_output.csv")
test_dataset = pd.read_csv("training.csv")

train_input_tensor = torch.tensor(training_dataset_input.values).to(torch.float32)
train_output_tensor = torch.tensor(training_dataset_output.values).to(torch.float32)
test_tensor = torch.tensor(test_dataset.values)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class PositiveOrNegative(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        # x.detach().apply_(lambda x : x * 100)
        x = F.sigmoid(self.fc1(x))
        return x
        
    def configure_optimizers(self):
        return torch.optim.SGD(model.parameters(), lr=0.01)

model = PositiveOrNegative()
print(model)

# Define your training data (input_data and target_data)
# Assuming you have your data loaded into PyTorch tensors or a DataLoader

# Define the loss function (binary cross-entropy loss for binary classification)
criterion = nn.BCELoss()

# Define the optimizer (e.g., stochastic gradient descent)
optimizer = model.configure_optimizers()

# Training loop (assuming you have your training loop set up)
for epoch in range(100000):
    optimizer.zero_grad()  # Zero the gradients
    output = model(train_input_tensor)  # Forward pass
    #output.detach().apply_(lambda x : sigmoid(x / 100))
    output.detach().apply_(lambda x : multi_step(x))
    loss = criterion(output, train_output_tensor)  # Calculate the loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if loss.item() > 0:
       with open("result.txt", "w") as file:
           file.write(str(output.detach().numpy()) + "\n" + str(train_output_tensor.numpy()))

    print(f"Epoch [{epoch + 1}/{1000000}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'model5.pth')

