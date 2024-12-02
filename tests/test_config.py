import unittest
import sys
import os
from pathlib import Path
import tempfile
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import (
    ProjectConfig,
    PathConfig,
    ModelConfig,
    TokenizerConfig,
    TrainingConfig,
    InferenceConfig
)

class TestConfiguration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = ProjectConfig()
        self.temp_dir = tempfile.mkdtemp()

    def test_default_initialization(self):
        """Test initialization with default values."""
        config = ProjectConfig()
        
        # Check if all components are initialized
        self.assertIsInstance(config.paths, PathConfig)
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.tokenizer, TokenizerConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        self.assertIsInstance(config.inference, InferenceConfig)

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        custom_model_config = ModelConfig(
            vocab_size=1000,
            hidden_size=768,  # Must be divisible by num_heads (12)
            num_layers=4,
            num_heads=12
        )
        
        config = ProjectConfig(model=custom_model_config)
        
        # Check if custom values are set
        self.assertEqual(config.model.vocab_size, 1000)
        self.assertEqual(config.model.hidden_size, 768)
        self.assertEqual(config.model.num_layers, 4)
        self.assertEqual(config.model.num_heads, 12)
        
        # Check if other configs use defaults
        self.assertEqual(
            config.tokenizer.vocab_size,
            TokenizerConfig().vocab_size
        )

    def test_path_resolution(self):
        """Test path resolution in PathConfig."""
        paths = self.config.paths
        
        # Check if paths are properly resolved
        self.assertTrue(paths.project_root.is_absolute())
        self.assertEqual(
            paths.data_dir,
            paths.project_root / "data"
        )
        self.assertEqual(
            paths.tokenizer_dir,
            paths.data_dir / "tokenizer"
        )

    def test_directory_creation(self):
        """Test automatic directory creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create config with custom root
            paths = PathConfig(project_root=Path(tmp_dir))
            config = ProjectConfig(paths=paths)
            
            # Check if directories were created
            self.assertTrue((Path(tmp_dir) / "data").exists())
            self.assertTrue((Path(tmp_dir) / "checkpoints").exists())
            self.assertTrue((Path(tmp_dir) / "logs").exists())
            self.assertTrue((Path(tmp_dir) / "data" / "tokenizer").exists())

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "config.json"
            
            # Modify some values
            self.config.model.vocab_size = 1000
            self.config.training.batch_size = 64
            self.config.inference.temperature = 0.8
            
            # Save config
            self.config.to_json(str(json_path))
            
            # Check if file exists
            self.assertTrue(json_path.exists())
            
            # Load config
            loaded_config = ProjectConfig.from_json(str(json_path))
            
            # Check if values match
            self.assertEqual(loaded_config.model.vocab_size, 1000)
            self.assertEqual(loaded_config.training.batch_size, 64)
            self.assertEqual(loaded_config.inference.temperature, 0.8)

    def test_config_validation(self):
        """Test configuration validation."""
        config = ProjectConfig()
        
        # Test model config constraints
        self.assertGreater(config.model.hidden_size, 0)
        self.assertGreater(config.model.num_layers, 0)
        self.assertGreater(config.model.num_heads, 0)
        
        # Test training config constraints
        self.assertGreater(config.training.batch_size, 0)
        self.assertGreater(config.training.learning_rate, 0)
        self.assertGreaterEqual(config.training.test_size, 0)
        self.assertLessEqual(config.training.test_size, 1)

    def test_special_tokens_initialization(self):
        """Test special tokens initialization in TokenizerConfig."""
        config = TokenizerConfig()
        
        # Check default special tokens
        self.assertIn('<pad>', config.special_tokens)
        self.assertIn('<unk>', config.special_tokens)
        self.assertIn('<sos>', config.special_tokens)
        self.assertIn('<eos>', config.special_tokens)
        
        # Check custom special tokens
        custom_tokens = {'<custom>': 0, '<special>': 1}
        config = TokenizerConfig(special_tokens=custom_tokens)
        self.assertEqual(config.special_tokens, custom_tokens)

    def test_config_immutability(self):
        """Test that critical config values cannot be changed after initialization."""
        config = ProjectConfig()
        
        # Attempt to modify paths after initialization
        with self.assertRaises(Exception):
            config.paths.project_root = Path("/different/path")
            
        # Ensure other values can be modified
        try:
            config.training.batch_size = 64
            config.model.hidden_size = 512
        except Exception as e:
            self.fail(f"Failed to modify mutable config values: {e}")

    def test_device_handling(self):
        """Test device configuration handling."""
        config = ProjectConfig()
        
        # Check if device is set correctly
        self.assertIn(config.training.device, ['cuda', 'cpu'])
        self.assertIn(config.inference.device, ['cuda', 'cpu'])
        
        # Test custom device setting
        config.training.device = 'cpu'
        config.inference.device = 'cpu'
        self.assertEqual(config.training.device, 'cpu')
        self.assertEqual(config.inference.device, 'cpu')

if __name__ == '__main__':
    unittest.main()