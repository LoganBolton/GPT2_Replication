class DataLoader:
    def __init__(self):
        self.block_size = 1024
        
    def chunk_data(block_size):
        dataset_path = "/home/log/Github/GPT2_Replication/input.txt"
        with open(dataset_path, 'r') as f:
            content = f.read()
        chunks = []

        for i in range(len(content)):
            chunk = content[i:i+block_size+1]
            chunks.append(chunk)


        return chunks