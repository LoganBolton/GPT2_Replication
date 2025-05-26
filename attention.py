import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt import GPTConfig  # or wherever your config class lives

class MultiHeadAttention:
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head        

    def forward(self, x):
        # x: [batch_size, seq_len, n_embd]
        return  # [batch_size, seq_len, n_embd]