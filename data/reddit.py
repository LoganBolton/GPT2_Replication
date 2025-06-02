import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import random
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import random

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Config
vocab_size = len(tokenizer)
block_size = 512
batch_size = 32  # Adjust based on your GPU memory
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RedditDataset(Dataset):
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


class RedditDataLoader:
    """Modern DataLoader wrapper for QA datasets"""
    
    def __init__(self, dataset_name="webis/tldr-17", 
                 train_split=0.9, 
                 batch_size=8, 
                 block_size=512,
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
        self.train_dataset = RedditDataset(
            data=full_data.select(train_indices),
            tokenizer=tokenizer,
            block_size=block_size,
            split='train'
        )
        
        self.val_dataset = RedditDataset(
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



class StreamingRedditDataset(IterableDataset):
    """
    Memory-efficient streaming dataset that loads data on-demand
    """
    
    def __init__(self, dataset_name, tokenizer, block_size, 
                 split='train', max_samples=None, shuffle_buffer=1000):
        """
        Args:
            dataset_name: HuggingFace dataset name
            tokenizer: Tokenizer to use
            block_size: Maximum sequence length
            split: 'train' or 'validation' 
            max_samples: Maximum number of samples to use (None = unlimited)
            shuffle_buffer: Size of shuffle buffer for randomization
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.split = split
        self.max_samples = max_samples
        self.shuffle_buffer = shuffle_buffer
        
    def __iter__(self):
        """
        This is where the magic happens - data is loaded one item at a time
        """
        # Load dataset in streaming mode - NO MEMORY USAGE YET!
        dataset = load_dataset(self.dataset_name, streaming=True)
        data_stream = dataset['train']
        
        # Optional: Skip some data for train/val split
        if self.split == 'validation':
            # Skip first 90% for validation set
            data_stream = data_stream.skip(int(0.9 * 1000000))  # Approximate
        elif self.split == 'train':
            # Take first 90% for training set  
            data_stream = data_stream.take(int(0.9 * 1000000))
            
        # Process samples one by one
        processed_count = 0
        buffer = []  # Small buffer for shuffling
        
        for item in data_stream:
            # Stop if we've reached our limit
            if self.max_samples and processed_count >= self.max_samples:
                break
                
            try:
                # Process this single item
                processed_item = self._process_item(item)
                if processed_item is not None:
                    buffer.append(processed_item)
                    
                    # When buffer is full, shuffle and yield items
                    if len(buffer) >= self.shuffle_buffer:
                        random.shuffle(buffer)
                        for buffered_item in buffer:
                            yield buffered_item
                            processed_count += 1
                        buffer = []  # Clear buffer
                        
            except Exception as e:
                # Skip problematic items
                print(f"Skipping item due to error: {e}")
                continue
        
        # Yield remaining items in buffer
        if buffer:
            random.shuffle(buffer)
            for buffered_item in buffer:
                yield buffered_item
    
    def _process_item(self, item):
        """
        Process a single data item - this is where ONE item gets tokenized
        """
        try:
            # Format the text (same as before)
            text = self._format_qa_pair(item)
            
            # Tokenize this ONE item
            tokens = self.tokenizer.encode(
                text, 
                truncation=True, 
                max_length=self.block_size,
                add_special_tokens=False
            )
            
            # Add BOS token
            tokens = [self.tokenizer.bos_token_id] + tokens
            
            # Skip if too short
            if len(tokens) < 10:
                return None
            
            # Pad or truncate to exact block_size
            if len(tokens) > self.block_size:
                tokens = tokens[:self.block_size]
            elif len(tokens) < self.block_size:
                pad_length = self.block_size - len(tokens)
                tokens = tokens + [self.tokenizer.pad_token_id] * pad_length
            
            # Convert to tensor
            tokens = torch.tensor(tokens, dtype=torch.long)
            
            # Return input and target (shifted by 1)
            return tokens[:-1], tokens[1:]
            
        except Exception as e:
            print(f"Error processing item: {e}")
            return None
    
    def _format_qa_pair(self, item):
        """Format a QA pair - same as before"""
        # Handle different possible field names
        instruction = item.get('instruction', item.get('prompt', item.get('question', '')))
        response = item.get('response', item.get('completion', item.get('answer', '')))
        
        # Create formatted text
        text = f"### USER: {instruction}\n\n### ASSISTANT: {response}"
        text += self.tokenizer.eos_token
        return text


class StreamingDataManager:
    """
    Manages streaming datasets and creates DataLoaders
    """
    
    def __init__(self, dataset_name="webis/tldr-17", 
                 batch_size=4, block_size=512, 
                 max_train_samples=100000, max_val_samples=5000):
        
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.block_size = block_size
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        
        print(f"Setting up streaming datasets for {dataset_name}")
        print(f"Max train samples: {max_train_samples}")
        print(f"Max val samples: {max_val_samples}")
        print(f"Batch Size: {batch_size}")
        
    def get_train_loader(self):
        """Get training data loader"""
        train_dataset = StreamingRedditDataset(
            dataset_name=self.dataset_name,
            tokenizer=tokenizer,
            block_size=self.block_size,
            split='train',
            max_samples=self.max_train_samples,
            shuffle_buffer=1000
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=0,  # Must be 0 for streaming datasets
            pin_memory=True if device == 'cuda' else False
        )
    
    def get_val_loader(self):
        """Get validation data loader"""
        val_dataset = StreamingRedditDataset(
            dataset_name=self.dataset_name,
            tokenizer=tokenizer,
            block_size=self.block_size,
            split='validation',
            max_samples=self.max_val_samples,
            shuffle_buffer=100  # Smaller buffer for validation
        )
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True if device == 'cuda' else False
        )
