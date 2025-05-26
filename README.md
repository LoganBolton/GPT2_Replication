# GPT2_Replication


## Chat Outline 

🧠 TODO.md – Build GPT-2 From Scratch
✅ Phase 0: Foundation Setup
 Understand GPT-2 at a system level: What are its key components? How does it differ from vanilla Transformer?

 Define architectural constants: n_layer, n_head, n_embd, vocab_size, block_size

 Create a minimal training loop placeholder (just skeleton, no actual training logic yet)

🔥 Phase 1: Core Model Components
 Implement MultiHeadSelfAttention

 Manually derive Q, K, V projections and their split across heads

 Apply causal masking (attention is NOT bidirectional)

 Think: How does masking enforce autoregression?

 Implement FeedForward (MLP block)

 GELU activation

 Layer norm before or after? Why does GPT-2 use pre-norm?

 Implement a single TransformerBlock

 Attention → residual → MLP → residual

 LayerNorm placement matters

 Stack N transformer blocks to create a GPT2Model (without output head)

🧩 Phase 2: Positionality & Tokens
 Implement TokenEmbedding + PositionalEmbedding

 Why is GPT-2 using learned positional embeddings? What does this imply about generalization?

 Sum token + position embeddings → feed into transformer stack

🧠 Phase 3: Language Modeling Objective
 Add a projection head: map final hidden states to vocab logits

 Implement loss function: nn.CrossEntropyLoss over shifted targets

 Think: Why do we shift the input vs target?

🔄 Phase 4: Sampling & Inference
 Implement autoregressive sampling (greedy → top-k → nucleus)

 How does causal masking affect token generation in practice?

 Think about temperature. Why does it matter?

 Clamp generation to a context window (sliding block_size)

🧪 Phase 5: Debugging Tools
 Visualize attention weights (per head, per layer)

 Compare output probabilities for similar inputs

 Optional: add hooks to inspect Q/K/V matrices

🧠 Phase 6: Go Deeper
 Implement caching for inference (kv-caching)

 Add rotary embeddings as an alternative to learned positions

 Train on toy data (e.g. "hello world", char-level Shakespeare)

