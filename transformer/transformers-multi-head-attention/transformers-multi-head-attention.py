import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_o: np.ndarray,
    num_heads: int
) -> np.ndarray:
    """
    Compute multi-head attention.
    """

    # Get dimensions
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Step 1: Linear projections
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    # Step 2: Split into multiple heads
    Q_heads = Q_proj.reshape(
        batch_size, seq_len, num_heads, d_k
    ).transpose(0, 2, 1, 3)

    K_heads = K_proj.reshape(
        batch_size, seq_len, num_heads, d_k
    ).transpose(0, 2, 1, 3)

    V_heads = V_proj.reshape(
        batch_size, seq_len, num_heads, d_k
    ).transpose(0, 2, 1, 3)

    # Step 3: Scaled Dot-Product Attention
    scores = np.matmul(
        Q_heads,
        K_heads.transpose(0, 1, 3, 2)
    )

    scores = scores / np.sqrt(d_k)

    attention_weights = softmax(scores, axis=-1)

    head_outputs = np.matmul(
        attention_weights,
        V_heads
    )

    # Step 4: Concatenate heads
    concat = head_outputs.transpose(
        0, 2, 1, 3
    ).reshape(
        batch_size,
        seq_len,
        d_model
    )

    # Step 5: Output projection
    output = concat @ W_o

    return output