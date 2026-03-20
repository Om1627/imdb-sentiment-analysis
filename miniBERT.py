import numpy as np

def softmax(x):
    """Stable softmax for the last dimension."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class LayerNorm:
    def __init__(self, d_model, eps=1e-12):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weights
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = np.dot(x, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = np.dot(x, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = np.dot(x, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        # scores = (Q @ K.T) / sqrt(d_k)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        attn = softmax(scores)
        context = np.matmul(attn, V) # [batch, heads, seq, d_k]

        # Concatenate heads
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        return np.dot(context, self.W_o)



class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Feed-forward weights
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.W2 = np.random.randn(d_ff, d_model) * 0.02

    def forward(self, x):
        # Sublayer 1: Attention + Residual + Norm
        attn_out = self.attention.forward(x)
        x = self.norm1.forward(x + attn_out)
        
        # Sublayer 2: Feed Forward + Residual + Norm
        ff_out = np.maximum(0, np.dot(x, self.W1)) # ReLU
        ff_out = np.dot(ff_out, self.W2)
        x = self.norm2.forward(x + ff_out)
        return x



class MiniBERTFromScratch:
    def __init__(self, vocab_size, max_seq_len, d_model, num_layers, num_heads, d_ff):
        # Embeddings
        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_emb = np.random.randn(max_seq_len, d_model) * 0.02
        
        # Encoder stack
        self.layers = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        
        # Classification Head
        self.classifier = np.random.randn(d_model, 2) * 0.02

    def forward(self, input_ids):
        # 1. Embedding Layer
        seq_len = input_ids.shape[1]
        x = self.token_emb[input_ids] + self.pos_emb[:seq_len]
        
        # 2. Transformer Layers
        for layer in self.layers:
            x = layer.forward(x)
            
        # 3. Pooling (BERT uses the first token [CLS] for classification)
        cls_output = x[:, 0, :]
        logits = np.dot(cls_output, self.classifier)
        return logits

# --- Usage Example ---
# vocab_size=30522 (Standard BERT), d_model=128, layers=2
model = MiniBERTFromScratch(30522, 512, 128, 2, 4, 512)

# Fake input: Batch of 1, Sequence of 5 tokens
input_data = np.array([[101, 2023, 2003, 2070, 102]]) # [CLS] this is good [SEP]
output = model.forward(input_data)

print("Output Logits (Positive/Negative):", output)
print("Predicted Class:", np.argmax(output))