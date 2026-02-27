import torch
from torch.utils.data import DataLoader
from tokenizer import SimpleTokenizer
from dataset import GPTDataset
from model import GPT

# Load text
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(base_dir, "data", "input.txt")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Build tokenizer
tokenizer = SimpleTokenizer(text)
tokens = tokenizer.encode(text)

block_size = 8
batch_size = 32

dataset = GPTDataset(tokens, block_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT(len(tokenizer.vocab)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(5):
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} | Loss: {loss.item()}")

def generate(model, start_text, tokenizer, max_new_tokens=20):
    model.eval()
    with torch.no_grad():
    
        tokens = tokenizer.encode(start_text)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

        for _ in range(max_new_tokens):
            tokens = tokens[:, -block_size:]
            logits = model(tokens)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

        generated_ids = tokens[0].tolist()
        return tokenizer.decode(generated_ids)

print("\n=== Text Generation ===")
output = generate(model, "Deep learning", tokenizer)
print(output)

torch.save(model.state_dict(), "gpt_model.pth")
print("\nModel saved as gpt_model.pth")