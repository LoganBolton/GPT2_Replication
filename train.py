from model import *
from data.shakespeare import *
from data.qadata import *
from data.reddit import *

torch.cuda.set_device(1)  # Sets GPU 1 as default
device = 'cuda:1'  # Be explicit about GPU 1

model = GPT()
model = model.to(device)  # Goes to GPU 1

# Add DataParallel with explicit device ordering
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with GPU 1 as master")
    # Put GPU 1 first in the device_ids list
    device_ids = [1, 0] + list(range(2, torch.cuda.device_count()))
    model = nn.DataParallel(model, device_ids=device_ids)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

max_iters = 100_000
# data_loader = RedditDataLoader(block_size=512, batch_size=32)
data_loader = QaDataLoader(block_size=512, batch_size=32)
train_loader, val_loader = data_loader.get_loaders()

# # data_loader = DataLoader()
# # splits = data_loader.get_splits()
# # splits = data_loader.get_loaders

# data_manager = StreamingDataManager(
#     dataset_name="webis/tldr-17",
#     batch_size=16,  # Much smaller
#     max_train_samples=100000  # Only use 100k samples instead of 1.5M
# )
# train_loader = data_manager.get_train_loader()
# val_loader = data_manager.get_val_loader()


print(f"Training on {device}")
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

step = 0
while step < max_iters:
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        y = y.view(B*T)
        # loss = F.cross_entropy(logits, y)
        loss = F.cross_entropy(logits, y, ignore_index=tokenizer.pad_token_id)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        
        # Evaluate and print both train and val loss every eval_interval steps
        if step % eval_interval == 0:
            losses = estimate_loss_streaming(model, train_loader, val_loader, eval_iters, device)
            print(f"Step {step}: Train Loss = {losses['train']:.4f}, Val Loss = {losses['val']:.4f}")
        # Print training loss every 200 steps
        elif step % 200 == 0:
            print(f"Step {step}: Train Loss = {loss.item():.4f}")
        
        
        if step >= max_iters:
            break

        if step % 5000 == 0:
            save_dir = f"checkpoints/gpt_{step}"
            torch.save(model.state_dict(), save_dir)
            print(f"Model saved to {save_dir}")

torch.save(model.state_dict(), 'gpt_model_final.pth')
print("Model saved to gpt_model.pth")