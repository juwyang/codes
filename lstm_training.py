import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed

# Set a seed for reproducibility
set_seed(42)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Generate pseudo data
class PseudoDataset(Dataset):
    def __init__(self, num_samples, seq_length, input_size):
        self.data = torch.randn(num_samples, seq_length, input_size)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Hyperparameters
input_size = 10
hidden_size = 64
num_layers = 2
output_size = 2
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Create dataset and dataloader
train_dataset = PseudoDataset(num_samples=10000, seq_length=50, input_size=input_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize the Accelerator
accelerator = Accelerator()

# Prepare the model, optimizer, and dataloader for distributed training
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_data, batch_labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        accelerator.backward(loss)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    accelerator.print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the model
accelerator.save(model.state_dict(), "lstm_model.pth")
accelerator.print("Training completed and model saved.")