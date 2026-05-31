import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    #compute the attention score
    scores= torch.matmul(Q,K.transpose(-2,-1))

    #scale by sqrt(d_k)
    d_k = K.size(-1)
    scores = scores / math.sqrt(d_k)
    #apply softmax
    attention_weights = F.softmax(scores, dim=-1)

    #step 4 weighted sum of value
    output = torch.matmul(attention_weights,V)
    return output