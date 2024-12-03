import re
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

class BPETokenizer:
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.token2id = {}
        self.id2token = {}
        
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3,
        }
        
        self.token2id.update(self.special_tokens)
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.next_token_id = len(self.special_tokens)

    def _add_token(self, token: str) -> int:
        if token not in self.token2id:
            self.token2id[token] = self.next_token_id
            self.id2token[self.next_token_id] = token
            self.next_token_id += 1
        return self.token2id[token]

    def build_vocab(self, corpus: str) -> None:
        # Initial character-level tokenization with progress bar
        print("Building initial vocabulary...")
        words = []
        for word in tqdm(corpus.split(), desc="Tokenizing words"):
            chars = list(word)
            # Add characters to vocabulary
            for c in chars:
                self._add_token(c)
            words.append(' '.join(chars) + ' </w>')
        
        self.vocab = Counter(words)
        
        # Calculate maximum merges needed
        max_merges = min(
            self.vocab_size - self.next_token_id,  # Available vocab space
            sum(len(word.split()) - 1 for word in self.vocab)  # Maximum possible merges
        )
        
        print(f"Learning {max_merges} BPE merges...")
        # Perform merges with progress bar
        pbar = tqdm(total=max_merges, desc="Learning BPE merges")
        while self.next_token_id < self.vocab_size:
            # Find most frequent pair
            pairs = defaultdict(int)
            for word, freq in self.vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[symbols[i], symbols[i + 1]] += freq
            
            if not pairs:
                break
                
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            merged_token = ''.join(best_pair)
            
            # Add to vocabulary
            self._add_token(merged_token)
            self.merges.append(best_pair)
            
            # Update vocabulary with merged tokens
            new_vocab = {}
            for word, freq in self.vocab.items():
                symbols = word.split()
                i = 0
                new_symbols = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                        new_symbols.append(merged_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                new_vocab[' '.join(new_symbols)] = freq
                
            self.vocab = new_vocab
            pbar.update(1)
            
        pbar.close()

    def encode(self, text: str, max_length: Optional[int] = None, padding: bool = False, truncation: bool = False) -> List[int]:
        # Start with special token
        tokens = ['<sos>']
        
        # Process each word
        for word in text.split():
            # Start with characters
            current = list(word)
            
            # Apply merges iteratively
            while True:
                # Try to apply each merge rule
                merged = False
                for old_pair in self.merges:
                    if merged:
                        break
                    
                    combined = ''.join(old_pair)
                    i = 0
                    while i < len(current) - 1:
                        if current[i] == old_pair[0] and current[i + 1] == old_pair[1]:
                            current[i:i + 2] = [combined]
                            merged = True
                        else:
                            i += 1
                            
                if not merged:
                    break
            
            # Add end of word token
            current.append('</w>')
            tokens.extend(current)
        
        # Add end of sequence token
        tokens.append('<eos>')
        
        # Convert to ids
        token_ids = [self.token2id.get(token, self.special_tokens['<unk>']) 
                    for token in tokens]
        
        # Handle length constraints
        if max_length is not None:
            if len(token_ids) > max_length and truncation:
                token_ids = token_ids[:max_length-1] + [self.special_tokens['<eos>']]
            elif len(token_ids) < max_length and padding:
                token_ids.extend([self.special_tokens['<pad>']] * 
                            (max_length - len(token_ids)))
        
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        current_word = []

        for id in token_ids:
            token = self.id2token.get(id, '<unk>')
            
            if token in self.special_tokens:
                continue
                
            if token == '</w>':
                tokens.append(''.join(current_word))
                current_word = []
            else:
                current_word.append(token)
        
        if current_word:
            tokens.append(''.join(current_word))
        
        return ' '.join(tokens).strip()

    def save(self, directory: str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        vocab_file = directory / 'vocab.json'
        with vocab_file.open('w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'token2id': self.token2id,
                'special_tokens': self.special_tokens
            }, f, ensure_ascii=False, indent=2)
        
        merges_file = directory / 'merges.txt'
        with merges_file.open('w', encoding='utf-8') as f:
            for pair in self.merges:
                f.write(f'{pair[0]} {pair[1]}\n')

    def load(self, directory: str) -> None:
        directory = Path(directory)
        
        vocab_file = directory / 'vocab.json'
        with vocab_file.open('r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.token2id = data['token2id']
            self.special_tokens = data['special_tokens']
            self.id2token = {int(v): k for k, v in self.token2id.items()}
            self.next_token_id = max(map(int, self.token2id.values())) + 1
        
        merges_file = directory / 'merges.txt'
        self.merges = []
        with merges_file.open('r', encoding='utf-8') as f:
            for line in f:
                token1, token2 = line.strip().split()
                self.merges.append((token1, token2))