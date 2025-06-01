from transformers import GPT2Tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from qadata import *

eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 1 # simple single head attention
n_layer = 6 # number of blocks
dropout = 0.2

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Config
vocab_size = len(tokenizer)
block_size = 512
batch_size = 32  # Adjust based on your GPU memory


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