import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words."""
        self.word_to_id ={
            self.pad_token : 0,
            self.unk_token : 1,
        self.bos_token : 2,
        self.eos_token : 3
        }

        #reverse mapping 
        self.id_to_word = {
            0: self.pad_token,
            1 : self.unk_token,
            2 : self.bos_token,
            3 : self.eos_token
        }
        #collect unique words
        unique_words = set()
        for text in texts:
            #lowercase + split
            words = text.lower().split()
            #add words to set
            unique_words.update(words)

        #sort the words aphabetically 
        sorted_words = sorted(unique_words)
        #assign IDs starting from 4
        current_id =4

        for word in sorted_words:
            self.word_to_id[word] = current_id
            self.id_to_word[current_id] =word

            current_id +=1

            #step 5: Store vocab size 
        self.vocab_size = current_id
        
            

    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # Lowercase + split
        words = text.lower().split()
        token_ids = [] 

        for word in words:

            #get ID if exists else UNK =1
            token_id = self.word_to_id.get(word,1)

            token_ids.append(token_id)
        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        
        words = []

        for token_id in ids:
            #get word if exists else <UNK>
            word = self.id_to_word.get(token_id, self.unk_token)
            words.append(word)
            
        return " ".join(words)
