import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformer import TinyTransformer

# 1) Load & tokenize on spaces
with open('train_data.text') as f:
    words = f.read().split()      # split on space

# 2) Build vocab
vocab = sorted(set(words))
stoi = {w:i for i,w in enumerate(vocab)}

# 3) Encode entire corpus
data = [stoi[w] for w in words]

# 4) Make sliding-window input→target pairs
block_size = 5
inputs  = [data[i : i+block_size]       for i in range(len(data)-block_size)]
targets = [data[i+1 : i+block_size+1]   for i in range(len(data)-block_size)]

class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

dataset = TextDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 5) Build model, loss, optimizer
vocab_size = len(vocab)
model      = TinyTransformer(vocab_size)
criterion  = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)

# 6) Simple training loop
for epoch in range(10):
    for xb, yb in dataloader:
        logits = model(xb)                          # (batch, seq_len, vocab_size)
        loss   = criterion(logits.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch:2d}  Loss: {loss.item():.4f}')

# 7) Save the trained model
torch.save(model.state_dict(), 'model.pth')