import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import random
from typing import Dict, List, Tuple, Optional

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Config
vocab_size = len(tokenizer)
block_size = 512
batch_size = 32  # Adjust based on your GPU memory
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class QADataset(Dataset):
    """Dataset for Q&A style data with proper formatting"""
    
    def __init__(self, data, tokenizer, block_size, split='train'):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        
        # Process and tokenize all examples
        for item in data:
            # Format the QA pair
            text = self._format_qa_pair(item)
            
            # Tokenize without adding special tokens automatically
            tokens = tokenizer.encode(
                text, 
                truncation=True, 
                max_length=block_size,
                add_special_tokens=False  # Don't add BOS automatically
            )
            
            # Manually add BOS at the beginning
            tokens = [tokenizer.bos_token_id] + tokens
            
            # Only keep examples that are long enough
            self.examples.append(tokens)
    
    def _format_qa_pair(self, item):
        """Format a single QA pair - just concatenate instruction and response"""
        instruction = item.get('instruction', '')
        context = item.get('context', '')
        response = item.get('response', '')
        
        text = f"### USER: {instruction}\n\n### Answer: {response}" 
        # Add EOS token for clear sequence boundaries
        text += tokenizer.eos_token
        
        return text
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Ensure all sequences are exactly block_size length
        if len(tokens) > self.block_size:
            # Truncate to block_size
            tokens = tokens[:self.block_size]
        elif len(tokens) < self.block_size:
            # Pad to block_size
            pad_length = self.block_size - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * pad_length
    
        # Convert to tensor
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # For language modeling, input is tokens[:-1], target is tokens[1:]
        return tokens[:-1], tokens[1:]


class QADataLoader:
    """Modern DataLoader wrapper for QA datasets"""
    
    def __init__(self, dataset_name="databricks/databricks-dolly-15k", 
                 train_split=0.9, 
                 batch_size=8, 
                 block_size=256,
                 num_workers=0):
        
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_workers = num_workers
        
        # Load dataset from HuggingFace
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        
        # The dolly dataset only has a 'train' split, so we need to create train/val
        full_data = dataset['train']
        
        # Shuffle and split
        indices = list(range(len(full_data)))
        random.shuffle(indices)
        
        split_idx = int(len(indices) * train_split)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Create datasets
        self.train_dataset = QADataset(
            data=full_data.select(train_indices),
            tokenizer=tokenizer,
            block_size=block_size,
            split='train'
        )
        
        self.val_dataset = QADataset(
            data=full_data.select(val_indices),
            tokenizer=tokenizer,
            block_size=block_size,
            split='val'
        )
        
        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device == 'cuda' else False
        )
    
    def get_batch(self, split='train'):
        """Get a single batch for compatibility with old code"""
        loader = self.train_loader if split == 'train' else self.val_loader
        
        # Get one batch from the iterator
        try:
            x, y = next(iter(loader))
        except StopIteration:
            # If iterator is exhausted, create a new one
            x, y = next(iter(loader))
        
        return x.to(device), y.to(device)
    
    def get_loaders(self):
        """Get the full DataLoaders for training loops"""
        return self.train_loader, self.val_loader

