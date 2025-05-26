import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt import GPTConfig  # or wherever your config class lives


batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2


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