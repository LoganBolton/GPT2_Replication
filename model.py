import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer

batch_size = 32 # how many independent sequences will we process in parallel?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 1 # simple single head attention
n_layer = 6 # number of blocks
dropout = 0.2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
vocab_size = len(tokenizer)
block_size = 256


class DataLoader:
    def __init__(self):
        self.inputs, self.outputs = self.chunk_data()

    def chunk_data(self):
        dataset_path = "input.txt"
        with open(dataset_path, 'r') as f:
            content = f.read()

        tokens = tokenizer.encode(content, add_special_tokens=False)
        print(f"Num Tokens: {len(tokens)}")
        print(f"Num of blocks = {len(tokens)//block_size}")
        inputs = []
        outputs = []

        for i in range(0, len(tokens) - block_size, block_size):
            x = tokens[i:i+block_size]
            inputs.append(x)

            y = tokens[i+1 : i+1+block_size]
            outputs.append(y)

        return inputs, outputs

    def get_splits(self):
        inputs, outputs = self.chunk_data()
        
        inputs = torch.tensor(inputs, dtype=torch.long)
        outputs = torch.tensor(outputs, dtype=torch.long)
        
        # Create train/val split (90% train, 10% val)
        n = len(inputs)
        split_idx = int(0.9 * n)
        
        train_inputs = inputs[:split_idx]
        train_outputs = outputs[:split_idx]
        
        val_inputs = inputs[split_idx:]
        val_outputs = outputs[split_idx:]
        
        return {
            'train': (train_inputs, train_outputs),
            'val': (val_inputs, val_outputs)
        }
        
    def get_batch(self, split_data):
        """Get a random batch from the split data"""
        inputs, outputs = split_data
        
        # Randomly sample batch_size indices
        batch_indices = torch.randint(0, len(inputs), (batch_size,))
        
        # Get the batched data
        x = inputs[batch_indices]
        y = outputs[batch_indices]
        
        return x.to(device), y.to(device)

    
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.head_size = n_embd // n_head

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        returns: (batch_size, seq_len, embed_dim)
        """
        # batches, time, channels
        B,T,C = x.shape
        # 1. Compute Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # 2. Compute attention scores
        # K = B,T,C
        # k_T = B,C,T
        k_T = k.transpose(1, 2)
        
        # qkt = B, T, T
        qkt = q @ k_T
        qkt = qkt * (self.head_size**-0.5) # scale by C

        mask = torch.tril(torch.ones(T, T)).to(dtype=torch.bool, device=qkt.device)

        qkt = qkt.masked_fill(mask==0, float('-inf'))

        # 3. Apply softmax (with masking if needed)
        soft_qkt = F.softmax(qkt, -1)
        
        # 4. Multiply scores with V
        y = soft_qkt @ v

        # 5. Return attended output
        return y

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd)
        )
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Head(n_embd) # feed in head size
        self.ffwd = FeedForward()
        
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
    def forward(self,x):
        res = x
        x = self.attn(self.layer_norm1(x))
        x += res
        
        res = x
        x = self.ffwd(self.layer_norm2(x))
        x += res
        return x
    
    
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_new_tokens = 100
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
    
    def forward(self, idx):
        # idx is the list of token IDs from the input
        B, T = idx.shape  # batch size, sequence length
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb # token embeddings + positional embeddings
        
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        
        logits = self.lm_head(x)

        return logits

    def generate(self, idx):
        for _ in range(self.max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # B, T+1
        return idx
    def generate_stream(self, idx, max_new_tokens=100):
        """Generate tokens one at a time, yielding each as it's produced"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Yield the new token
            yield idx_next

    
# Initialize everything
data_loader = DataLoader()
splits = data_loader.get_splits()
model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)


####### TRAIN #########
# print(f"Training on {device}")
# print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
# for iter in range(max_iters):
#     x, y = data_loader.get_batch(splits['train'])
#     logits = model(x)
    
#     B, T, C = logits.shape
#     logits = logits.view(B*T, C)
#     y = y.view(B*T)
#     loss = F.cross_entropy(logits, y)
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     if iter % 100 == 0:
#         print(f"Iter {iter}: Loss = {loss.item():.4f}")

# torch.save(model.state_dict(), 'gpt_model.pth')
# print("Model saved to gpt_model.pth")


####### INFERENCE #########
model.load_state_dict(torch.load('gpt_model.pth', map_location=device))
model.eval()

prompt = "Hello"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

print(prompt, end='', flush=True)
with torch.no_grad():
    for token_id in model.generate_stream(input_ids, max_new_tokens=1000):
        if token_id.dim() > 0:  # It's a single token
            token = tokenizer.decode(token_id[0], skip_special_tokens=True)
            print(token, end='', flush=True)

# with torch.no_grad():
#     generated = model.generate(input_ids)

# text = tokenizer.decode(generated[0], skip_special_tokens=True)
# print(text)