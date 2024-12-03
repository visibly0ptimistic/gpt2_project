import re
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path

class BPETokenizer:
    def __init__(self, vocab_size: int = 5000):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size including special tokens
        """
        self.vocab_size = vocab_size
        self.vocab = {}  # word -> frequency mapping
        self.merges = []  # list of merge rules
        self.token2id = {}  # token -> id mapping
        self.id2token = {}  # id -> token mapping
        
        # Initialize special tokens
        self.special_tokens = {
            '<pad>': 0,    # Padding token
            '<unk>': 1,    # Unknown token
            '<sos>': 2,    # Start of sequence
            '<eos>': 3,    # End of sequence
        }
        
        # Initialize token2id with special tokens
        self.token2id.update(self.special_tokens)
        self.id2token = {v: k for k, v in self.token2id.items()}
        
        # Track next available token ID
        self.next_token_id = len(self.special_tokens)

    def _add_token(self, token: str) -> int:
        """
        Add a token to the vocabulary and return its ID.
        
        Args:
            token: Token to add
            
        Returns:
            Token ID
        """
        if token not in self.token2id:
            self.token2id[token] = self.next_token_id
            self.id2token[self.next_token_id] = token
            self.next_token_id += 1
        return self.token2id[token]

    def build_vocab(self, corpus: str) -> None:
        """
        Build vocabulary from text corpus using BPE algorithm.
        
        Args:
            corpus: Input text corpus
        """
        # Tokenize words into characters
        words = [' '.join(word) + ' </w>' for word in corpus.split()]
        self.vocab = Counter(words)

        # Continue merging pairs until vocab_size is reached
        while self.next_token_id < self.vocab_size:
            pairs = defaultdict(int)
            
            # Count pairs of characters/tokens in corpus
            for word, freq in self.vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq

            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Add merged token to vocabulary
            merged_token = ''.join(best_pair)
            self._add_token(merged_token)
            
            # Save merge rule
            self.merges.append(best_pair)

            # Apply merge to vocabulary
            new_vocab = {}
            for word, freq in self.vocab.items():
                new_word = word.replace(f"{best_pair[0]} {best_pair[1]}", merged_token)
                new_vocab[new_word] = freq
            self.vocab = new_vocab

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text
            max_length: Maximum sequence length (including special tokens)
            
        Returns:
            List of token IDs
        """
        # Start with SOS token
        tokens = ['<sos>']
        
        # Tokenize each word
        words = text.split()
        for word in words:
            word = ' '.join(word) + ' </w>'
            
            # Apply merges
            for merge in self.merges:
                word = re.sub(r' '.join(merge), ''.join(merge), word)
            
            # Add resulting tokens
            tokens.extend(word.split())

        # Add EOS token
        tokens.append('<eos>')
        
        # Convert tokens to IDs
        token_ids = [self.token2id.get(token, self.special_tokens['<unk>']) 
                    for token in tokens]
        
        # Handle sequence length
        if max_length is not None:
            if len(token_ids) > max_length:
                # Truncate, keeping SOS and EOS
                token_ids = token_ids[:max_length-1] + [self.special_tokens['<eos>']]
            else:
                # Pad to max_length
                token_ids.extend([self.special_tokens['<pad>']] * 
                               (max_length - len(token_ids)))
        
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        # Convert IDs to tokens and filter special tokens
        tokens = []
        current_word = []

        for id in token_ids:
            token = self.id2token.get(id, '<unk>')
            
            # Skip special tokens
            if token in self.special_tokens:
                continue
                
            # Handle word endings
            if token.endswith('</w>'):
                # Add token without </w> to current word
                current_word.append(token[:-4])
                # Add completed word to tokens
                tokens.append(''.join(current_word))
                current_word = []
            else:
                current_word.append(token)
        
        # Handle any remaining tokens in current_word
        if current_word:
            tokens.append(''.join(current_word))
        
        # Join words and clean up whitespace
        text = ' '.join(tokens)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def save(self, directory: str) -> None:
        """
        Save tokenizer vocabulary and merges to files.
        
        Args:
            directory: Directory to save files
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        vocab_file = directory / 'vocab.json'
        with vocab_file.open('w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'token2id': self.token2id,
                'special_tokens': self.special_tokens
            }, f, ensure_ascii=False, indent=2)
        
        # Save merges
        merges_file = directory / 'merges.txt'
        with merges_file.open('w', encoding='utf-8') as f:
            for pair in self.merges:
                f.write(f'{pair[0]} {pair[1]}\n')

    def load(self, directory: str) -> None:
        """
        Load tokenizer vocabulary and merges from files.
        
        Args:
            directory: Directory containing saved files
        """
        directory = Path(directory)
        
        # Load vocabulary
        vocab_file = directory / 'vocab.json'
        with vocab_file.open('r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.token2id = data['token2id']
            self.special_tokens = data['special_tokens']
            self.id2token = {int(v): k for k, v in self.token2id.items()}
            self.next_token_id = max(map(int, self.token2id.values())) + 1
        
        # Load merges
        merges_file = directory / 'merges.txt'
        self.merges = []
        with merges_file.open('r', encoding='utf-8') as f:
            for line in f:
                token1, token2 = line.strip().split()
                self.merges.append((token1, token2))