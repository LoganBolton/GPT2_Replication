import torch
import torch.nn as nn
import torch.nn.functional as F
from qadata import *

eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 1024
n_heads = 16 # simple single head attention
n_layer = 12 # number of blocks
dropout = 0.1

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Config
vocab_size = len(tokenizer)
block_size = 512
batch_size = 16  # Adjust based on your GPU memory

    
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.head_size = head_size

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

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_heads
        self.attn = MultiHeadAttention(n_heads=n_heads, head_size=head_size)
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

def estimate_loss(model, train_loader, val_loader, eval_iters, device):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        loader_iter = iter(loader)
        
        for k in range(eval_iters):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                # If we run out of data, restart the iterator
                loader_iter = iter(loader)
                x, y = next(loader_iter)
                
            x = x.to(device)
            y = y.to(device)
            
            with torch.no_grad():
                logits = model(x)
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                y = y.view(B*T)
                loss = F.cross_entropy(logits, y)
                losses[k] = loss.item()
        
        out[split] = losses.mean()
    
    model.train()
    return out