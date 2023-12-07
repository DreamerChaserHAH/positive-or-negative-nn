import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class PositiveOrNegative(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        # x.detach().apply_(lambda x : x * 100)
        x = F.sigmoid(self.fc1(x))
        return x
    
    def predict(self, value):
        output = self(value)
        return output.item()

    def configure_optimizers(self):
        return torch.optim.SGD(model.parameters(), lr=0.003)

# Load the pre-trained model
model = PositiveOrNegative()
model.load_state_dict(torch.load('model5.pth'))
model.eval()

train_input_tensor = torch.tensor([0]).to(torch.float32)
print(model.predict(train_input_tensor))
