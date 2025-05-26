from transformers import GPT2Tokenizer

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