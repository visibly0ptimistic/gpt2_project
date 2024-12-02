# GPT-2 Implementation

A clean, modular implementation of GPT-2 in PyTorch with byte-pair encoding (BPE) tokenization.

## Project Structure

```bash
gpt2_project/
├── README.md
├── checkpoints/        # Model checkpoints and saved states
├── data/              # Dataset and tokenizer files
│   ├── raw_dataset.txt
│   ├── train.pt
│   └── val.pt
├── scripts/           # Training and utility scripts
│   ├── inference.py   # Text generation script
│   ├── preprocess.py  # Data preprocessing
│   ├── test.py       # Testing utilities
│   └── train.py      # Training script
└── src/              # Core model implementation
    ├── config.py     # Configuration management
    ├── model.py      # GPT-2 model architecture
    └── tokenizer.py  # BPE tokenizer implementation
```

## Features

- Clean, modular implementation of GPT-2
- Byte-pair encoding (BPE) tokenization
- Configurable model size and architecture
- Training with gradient accumulation
- Text generation with top-k and nucleus sampling
- Weights & Biases integration for experiment tracking
- Comprehensive configuration management
- Interactive text generation interface

## Requirements

```bash
torch>=1.8.0
transformers
wandb
tqdm
sklearn
```

## Usage

### Data Preprocessing

Prepare your training data and preprocess it:

```bash
python scripts/preprocess.py
```

### Training

Train the model with default parameters:

```bash
python scripts/train.py
```

To customize training parameters, modify the configuration in `src/config.py` or create a custom JSON config file:

```python
from src.config import ProjectConfig

# Load custom config
config = ProjectConfig.from_json('my_config.json')

# Or modify default config
config = ProjectConfig()
config.training.batch_size = 64
config.training.learning_rate = 1e-4
```

### Text Generation

Generate text using the trained model:

```bash
# Interactive mode
python scripts/inference.py --interactive

# Single generation
python scripts/inference.py --prompt "Your prompt here" --max_length 100
```

Generation parameters:

- `--prompt`: Input text prompt
- `--max_length`: Maximum sequence length
- `--temperature`: Sampling temperature (default: 0.7)
- `--top_k`: Top-k sampling parameter (default: 50)
- `--top_p`: Nucleus sampling parameter (default: 0.9)
- `--num_sequences`: Number of sequences to generate (default: 1)

### Configuration

The project uses a hierarchical configuration system defined in `src/config.py`:

- `PathConfig`: Project paths and file locations
- `ModelConfig`: Model architecture parameters
- `TokenizerConfig`: Tokenizer settings
- `TrainingConfig`: Training hyperparameters
- `InferenceConfig`: Text generation parameters

Create a custom configuration:

```python
from src.config import ProjectConfig

config = ProjectConfig()
config.to_json('config.json')  # Save configuration
```

## Model Architecture

The implementation follows the original GPT-2 architecture:

- Token and position embeddings
- Multiple transformer layers with:
  - Multi-head self-attention
  - Layer normalization
  - Feed-forward neural network
- Weight tying between input embeddings and output layer

Key parameters (default configuration):

- Vocabulary size: 5000
- Hidden size: 768
- Number of layers: 12
- Number of attention heads: 12
- Maximum sequence length: 512

## Training Features

- Gradient accumulation for larger effective batch sizes
- Linear learning rate warmup
- Gradient clipping
- Model checkpointing
- Validation evaluation
- Weights & Biases integration for experiment tracking

## Text Generation Features

- Temperature-controlled sampling
- Top-k filtering
- Nucleus (top-p) sampling
- Interactive generation mode
- Batch generation support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
