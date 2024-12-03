import re
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path

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
        words = [' '.join(word) + ' </w>' for word in corpus.split()]
        self.vocab = Counter(words)

        while self.next_token_id < self.vocab_size:
            pairs = defaultdict(int)
            
            for word, freq in self.vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            merged_token = ''.join(best_pair)
            self._add_token(merged_token)
            self.merges.append(best_pair)

            new_vocab = {}
            for word, freq in self.vocab.items():
                pattern = re.escape(best_pair[0]) + r'\s+' + re.escape(best_pair[1])
                new_word = re.sub(pattern, merged_token, word)
                new_vocab[new_word] = freq
            self.vocab = new_vocab

    def encode(self, text: str, max_length: Optional[int] = None, padding: bool = False, truncation: bool = False) -> List[int]:
        tokens = ['<sos>']
        
        words = text.split()
        for word in words:
            word = ' '.join(word) + ' </w>'
            
            for merge in self.merges:
                pattern = re.escape(merge[0]) + r'\s+' + re.escape(merge[1])
                word = re.sub(pattern, ''.join(merge), word)
            
            tokens.extend(word.split())

        tokens.append('<eos>')
        
        token_ids = [self.token2id.get(token, self.special_tokens['<unk>']) 
                    for token in tokens]
        
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
                
            if token.endswith('</w>'):
                current_word.append(token[:-4])
                tokens.append(''.join(current_word))
                current_word = []
            else:
                current_word.append(token)
        
        if current_word:
            tokens.append(''.join(current_word))
        
        text = ' '.join(tokens)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

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