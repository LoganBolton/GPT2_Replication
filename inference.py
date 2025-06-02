from model import *

model = GPT().to(device)

####### INFERENCE #########
weights = "checkpoints/gpt_5000"
model.load_state_dict(torch.load(weights, map_location=device))
model.eval()

prompt = "### USER: What does it mean to be human\n\n### Answer: "

# print(prompt, end='', flush=True)
# with torch.no_grad():
#     for token_id in model.generate_stream(input_ids, max_new_tokens=1000):
#         if token_id.dim() > 0:  # It's a single token
#             token = tokenizer.decode(token_id[0], skip_special_tokens=True)
#             print(token, end='', flush=True)

# Collect all token_ids into a list
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

print(prompt, end='', flush=True)
with torch.no_grad():
    for token_id in model.generate_stream(input_ids, max_new_tokens=1000):
        token_str = tokenizer.decode(token_id[0], skip_special_tokens=True)
        print(token_str, end='', flush=True)


# with torch.no_grad():
#     generated = model.generate(input_ids)

# text = tokenizer.decode(generated[0], skip_special_tokens=True)
# print(text)