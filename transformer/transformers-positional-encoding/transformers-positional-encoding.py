import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    #creating empty matrix
    pe= np.zeros((seq_length,d_model))

    #create position column vector
    position = np.arange(seq_length).reshape(seq_length,1)

    #compute scaling terms
    div_term = np.exp(np.arange(0,d_model,2)*(-np.log(10000.0)/d_model))

    #apply sine to even columns 
    pe[:,0::2]=np.sin(position *div_term)

    #apply cosine to odd columns
    pe[:,1::2] = np.cos(position *div_term)

    return pe
