import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer

batch_size = 32 # how many independent sequences will we process in parallel?
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


class DataLoader:
    def __init__(self):
        self.block_size = 1024
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.inputs, self.outputs = self.chunk_data(self.block_size)
        


    def chunk_data(self, block_size):
        dataset_path = "/home/log/Github/GPT2_Replication/input.txt"
        with open(dataset_path, 'r') as f:
            content = f.read()

        tokens = self.tokenizer.encode(content, add_special_tokens=False)
        print(f"Num Tokens: {len(tokens)}")
        print(f"Num of blocks = {len(tokens)//block_size}")
        inputs = []
        outputs = []

        for i in range(0, len(tokens) - block_size, block_size):
            x = tokens[i:i+block_size+1]
            inputs.append(x)

            y = tokens[i+1 : i+1+block_size+1]
            outputs.append(y)

        return inputs, outputs
    
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
        self.fward = FeedForward()
    def forward(self,x):
        x = self.attn(x)
        x = self.fward(x)
        return x
    
    
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        return

    def generate(self):
        return