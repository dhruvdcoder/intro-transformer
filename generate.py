import torch
from transformer import TinyTransformer

# Rebuild vocab from training data
with open('train_data.text') as f:
    words = f.read().split()

vocab = sorted(set(words))
stoi  = {w:i for i,w in enumerate(vocab)}
itos  = {i:w for w,i in stoi.items()}

# Load model
vocab_size = len(vocab)
model = TinyTransformer(vocab_size)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Prompt and tokenize
prompt = 'hello world'
input_ids = torch.tensor([[stoi[w] for w in prompt.split()]], dtype=torch.long)

# Generate up to 10 new tokens
out_ids = model.decode(input_ids, max_length=10)

# Convert tokens back to words
tokens = [itos[idx] for idx in out_ids[0].tolist()]
print('Generated:', ' '.join(tokens))
