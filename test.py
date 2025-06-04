# Test the fixed dataset loading
# Run this in your notebook to verify the fix worked

from data.reddit import StreamingDataManager
import torch

# Create a small test data manager
data_manager = StreamingDataManager(
    dataset_name="webis/tldr-17",
    batch_size=4,  # Small batch for testing
    max_train_samples=100,  # Just 100 samples for quick test
    max_val_samples=50
)

train_loader = data_manager.get_train_loader()

print("=== TESTING FIXED DATASET ===")

# Test first few batches
for batch_idx, (x, y) in enumerate(train_loader):
    if batch_idx >= 3:  # Just test 3 batches
        break
    
    print(f"\n--- Batch {batch_idx} ---")
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")
    
    # Check for diversity - count unique sequences
    unique_sequences = len(set(tuple(seq.tolist()) for seq in x))
    print(f"Unique sequences in batch: {unique_sequences}/{x.shape[0]}")
    
    # Check padding ratio
    pad_token_id = 2  # Assuming this is still the pad token
    total_tokens = x.numel()
    pad_tokens = (x == pad_token_id).sum().item()
    pad_ratio = pad_tokens / total_tokens
    print(f"Padding ratio: {pad_ratio:.2%}")
    
    # Check token diversity
    unique_tokens = torch.unique(x).shape[0]
    print(f"Unique tokens in batch: {unique_tokens}")
    
    # Look at first sequence in detail
    seq = x[0]
    non_pad_tokens = seq[seq != pad_token_id]
    print(f"Non-pad tokens in first sequence: {len(non_pad_tokens)}")
    print(f"First 20 tokens: {seq[:20].tolist()}")
    
    # Try to decode if possible
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        decoded = tokenizer.decode(non_pad_tokens[:100].tolist(), skip_special_tokens=False)
        print(f"Decoded text sample: '{decoded[:200]}...'")
    except Exception as e:
        print(f"Decoding error: {e}")

print("\n=== EXPECTED RESULTS ===")
print("✅ Good signs:")
print("- Unique sequences should be close to batch_size (3-4 out of 4)")
print("- Padding ratio should be reasonable (30-70%)")
print("- Unique tokens should be much higher (hundreds/thousands)")
print("- Decoded text should show actual Reddit post content")
print("- No repeated identical sequences across batches")

print("\n❌ Bad signs:")
print("- All sequences identical")
print("- >90% padding")
print("- <50 unique tokens")
print("- Decoded text shows templates or gibberish")