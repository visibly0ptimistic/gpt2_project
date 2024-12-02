import unittest
import torch
import sys
import os
from pathlib import Path
import tempfile

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model import GPT2, GPT2Config, MultiHeadAttention, TransformerBlock

class TestGPT2Components(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = GPT2Config(
            vocab_size=100,
            max_position_embeddings=32,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            intermediate_size=128
        )
        self.batch_size = 2
        self.seq_length = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_attention_shapes(self):
        """Test MultiHeadAttention output shapes."""
        attention = MultiHeadAttention(self.config).to(self.device)
        hidden_states = torch.randn(
            self.batch_size,
            self.seq_length,
            self.config.hidden_size,
            device=self.device
        )
        
        # Test forward pass
        output = attention(hidden_states)
        
        # Check output shape
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_length, self.config.hidden_size)
        )

    def test_attention_mask(self):
        """Test attention masking."""
        attention = MultiHeadAttention(self.config).to(self.device)
        hidden_states = torch.randn(
            self.batch_size,
            self.seq_length,
            self.config.hidden_size,
            device=self.device
        )
        
        # Create attention mask (mask out second half of sequence)
        attention_mask = torch.ones(
            (self.batch_size, 1, 1, self.seq_length),
            device=self.device
        )
        attention_mask[:, :, :, self.seq_length//2:] = 0
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Get outputs with and without mask
        output_masked = attention(hidden_states, attention_mask)
        output_unmasked = attention(hidden_states)
        
        # Check that outputs are different
        self.assertFalse(torch.allclose(output_masked, output_unmasked))

    def test_transformer_block(self):
        """Test TransformerBlock functionality."""
        block = TransformerBlock(self.config).to(self.device)
        hidden_states = torch.randn(
            self.batch_size,
            self.seq_length,
            self.config.hidden_size,
            device=self.device
        )
        
        # Test forward pass
        output = block(hidden_states)
        
        # Check output shape
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_length, self.config.hidden_size)
        )
        
        # Test that output is different from input
        self.assertFalse(torch.allclose(output, hidden_states))

    def test_position_embeddings(self):
        """Test position embeddings."""
        model = GPT2(self.config).to(self.device)
        input_ids = torch.randint(
            0,
            self.config.vocab_size,
            (self.batch_size, self.seq_length),
            device=self.device
        )
        
        # Get position IDs
        position_ids = model.get_position_ids(input_ids)
        
        # Check shape
        self.assertEqual(
            position_ids.shape,
            (self.batch_size, self.seq_length)
        )
        
        # Check values
        expected_positions = torch.arange(self.seq_length, device=self.device)
        self.assertTrue(torch.all(position_ids[0] == expected_positions))

    def test_gpt2_forward(self):
        """Test full GPT2 model forward pass."""
        model = GPT2(self.config).to(self.device)
        input_ids = torch.randint(
            0,
            self.config.vocab_size,
            (self.batch_size, self.seq_length),
            device=self.device
        )
        
        # Test forward pass
        logits, hidden_states = model(input_ids)
        
        # Check shapes
        self.assertEqual(
            logits.shape,
            (self.batch_size, self.seq_length, self.config.vocab_size)
        )
        self.assertEqual(
            hidden_states.shape,
            (self.batch_size, self.seq_length, self.config.hidden_size)
        )

    def test_gpt2_generation(self):
        """Test text generation functionality."""
        model = GPT2(self.config).to(self.device)
        input_ids = torch.randint(
            0,
            self.config.vocab_size,
            (self.batch_size, 5),  # Start with short sequence
            device=self.device
        )
        max_length = 10
        
        # Test generation with different settings
        generations = {
            'basic': model.generate(input_ids, max_length=max_length),
            'temp': model.generate(input_ids, max_length=max_length, temperature=0.7),
            'top_k': model.generate(input_ids, max_length=max_length, top_k=5),
            'top_p': model.generate(input_ids, max_length=max_length, top_p=0.9),
        }
        
        for name, output in generations.items():
            # Check output shape
            self.assertEqual(
                output.shape[1],
                max_length,
                f"Generated sequence length mismatch for {name}"
            )
            # Check start matches input
            self.assertTrue(
                torch.all(output[:, :5] == input_ids),
                f"Generated sequence doesn't match input for {name}"
            )

    def test_save_load(self):
        """Test model saving and loading."""
        model = GPT2(self.config)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save model
            model.save_pretrained(tmp_dir)
            
            # Check if files exist
            self.assertTrue(Path(tmp_dir, 'config.json').exists())
            self.assertTrue(Path(tmp_dir, 'pytorch_model.bin').exists())
            
            # Load model
            loaded_model = GPT2.from_pretrained(tmp_dir)
            
            # Check if configs match
            self.assertEqual(
                vars(model.config),
                vars(loaded_model.config)
            )
            
            # Check if weights match
            for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.all(p1 == p2))

    def test_weight_tying(self):
        """Test if embedding weights are properly tied."""
        model = GPT2(self.config)
        
        # Check if embedding and output weights are the same object
        self.assertIs(
            model.token_embeddings.weight,
            model.head.weight
        )
        
        # Modify embedding and check if output weight changes
        original_weight = model.token_embeddings.weight.clone()
        model.token_embeddings.weight.data += 1
        self.assertTrue(
            torch.all(model.head.weight == model.token_embeddings.weight)
        )
        self.assertFalse(
            torch.all(model.head.weight == original_weight)
        )

if __name__ == '__main__':
    unittest.main()