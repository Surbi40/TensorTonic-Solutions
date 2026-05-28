import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    #get key dimension
    d_k = K.size(-1)

    #compute attention score 
    scores = torch.matmul(Q,K.transpose(-2,-1))

    #scale score
    scores= scores / math.sqrt(d_k)

    #apply softmax
    attention_weights = F.softmax(scores, dim=-1)

    #weighted sum of values
    output= torch.matmul(attention_weights, V)

    return output
