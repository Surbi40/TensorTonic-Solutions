import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    #handling empty valaue
    if len(seqs)==0:
        return numpy.empty((0,0),dtype=int)

    #determine max length
    if max_len is None:
        max_len=max(len(seq) for seq in seqs)

    #store padded sequence
    padded = []
    #process each sequence
    for seq in seqs:
        seq=seq[:max_len]
        padding_needed = max_len - len(seq)

        padded_seq = seq + [pad_value] * padding_needed

        padded.append(padded_seq)
    return np.array(padded,dtype=int)
        