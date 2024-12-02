import unittest
import sys
import os
from pathlib import Path
import tempfile

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.tokenizer import BPETokenizer

class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tokenizer = BPETokenizer(vocab_size=100)
        self.test_corpus = "hello world hello machine learning"
        self.tokenizer.build_vocab(self.test_corpus)

    def test_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = BPETokenizer(vocab_size=50)
        
        # Check special tokens
        self.assertEqual(tokenizer.special_tokens['<pad>'], 0)
        self.assertEqual(tokenizer.special_tokens['<unk>'], 1)
        self.assertEqual(tokenizer.special_tokens['<sos>'], 2)
        self.assertEqual(tokenizer.special_tokens['<eos>'], 3)
        
        # Check initial token mappings
        self.assertEqual(len(tokenizer.token2id), 4)
        self.assertEqual(len(tokenizer.id2token), 4)

    def test_vocab_building(self):
        """Test vocabulary building."""
        # Check vocab size
        self.assertLess(len(self.tokenizer.token2id), self.tokenizer.vocab_size)
        
        # Check merges
        self.assertGreater(len(self.tokenizer.merges), 0)
        
        # Check if common subwords are merged
        merged_token = ''.join(self.tokenizer.merges[0])
        self.assertIn(merged_token, self.tokenizer.token2id)

    def test_encoding(self):
        """Test text encoding."""
        text = "hello world"
        encoded = self.tokenizer.encode(text)
        
        # Check if output is a list of integers
        self.assertTrue(all(isinstance(x, int) for x in encoded))
        
        # Check if special tokens are added
        self.assertEqual(encoded[0], self.tokenizer.special_tokens['<sos>'])
        self.assertEqual(encoded[-1], self.tokenizer.special_tokens['<eos>'])
        
        # Test max length handling
        max_length = 5
        encoded_truncated = self.tokenizer.encode(text, max_length=max_length)
        self.assertEqual(len(encoded_truncated), max_length)

    def test_decoding(self):
        """Test token decoding."""
        original_text = "hello world"
        encoded = self.tokenizer.encode(original_text)
        decoded = self.tokenizer.decode(encoded)
        
        # Check if decoding removes special tokens
        self.assertNotIn('<sos>', decoded)
        self.assertNotIn('<eos>', decoded)
        self.assertNotIn('<pad>', decoded)
        
        # Check if decoded text matches original (ignoring case and whitespace)
        self.assertEqual(decoded.strip().lower(), original_text.strip().lower())

    def test_save_load(self):
        """Test saving and loading tokenizer state."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save tokenizer
            self.tokenizer.save(tmp_dir)
            
            # Check if files are created
            self.assertTrue(Path(tmp_dir, 'vocab.json').exists())
            self.assertTrue(Path(tmp_dir, 'merges.txt').exists())
            
            # Load into new tokenizer
            new_tokenizer = BPETokenizer(vocab_size=100)
            new_tokenizer.load(tmp_dir)
            
            # Check if states match
            self.assertEqual(self.tokenizer.vocab, new_tokenizer.vocab)
            self.assertEqual(self.tokenizer.token2id, new_tokenizer.token2id)
            self.assertEqual(self.tokenizer.merges, new_tokenizer.merges)

    def test_unknown_tokens(self):
        """Test handling of unknown tokens."""
        unknown_text = "xyz123"  # Tokens not in training corpus
        encoded = self.tokenizer.encode(unknown_text)
        
        # Check if unknown tokens are mapped to <unk>
        self.assertTrue(self.tokenizer.special_tokens['<unk>'] in encoded)
        
        decoded = self.tokenizer.decode(encoded)
        # Check if text can be decoded (even if not matching original)
        self.assertIsInstance(decoded, str)

    def test_empty_input(self):
        """Test handling of empty input."""
        encoded = self.tokenizer.encode("")
        
        # Should still include special tokens
        self.assertEqual(len(encoded), 2)  # <sos> and <eos>
        self.assertEqual(encoded[0], self.tokenizer.special_tokens['<sos>'])
        self.assertEqual(encoded[1], self.tokenizer.special_tokens['<eos>'])
        
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded.strip(), "")

if __name__ == '__main__':
    unittest.main()