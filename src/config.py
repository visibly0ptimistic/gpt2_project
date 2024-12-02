from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import torch
import json

@dataclass
class PathConfig:
    """Enhanced configuration for project paths."""
    # Base paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    checkpoints_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "checkpoints")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    # Dataset paths
    datasets: Dict[str, Path] = field(default_factory=lambda: {
        'openwebtext': Path(__file__).parent.parent / "data/openwebtext",
        'books': Path(__file__).parent.parent / "data/books",
        'custom': Path(__file__).parent.parent / "data/custom"
    })
    
    # Processed data paths
    processed_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data/processed")
    train_dataset_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data/processed/train.pt")
    val_dataset_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data/processed/val.pt")
    test_dataset_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data/processed/test.pt")
    
    # Tokenizer paths
    tokenizer_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data/tokenizer")
    vocab_file: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data/tokenizer/vocab.json")
    merges_file: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data/tokenizer/merges.txt")
    
    def __post_init__(self):
        """Create all necessary directories."""
        try:
            # Create base directories first
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Then create subdirectories
            for path in [self.processed_dir, self.tokenizer_dir, *self.datasets.values()]:
                path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create directory structure: {str(e)}")

@dataclass
class ModelConfig:
    """Enhanced configuration for model architecture."""
    vocab_size: int = 50257  # Standard GPT-2 vocabulary size
    max_position_embeddings: int = 1024
    hidden_size: int = 768  # d_model in transformer terminology
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072  # 4 * hidden_size is standard
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    use_cache: bool = True
    gradient_checkpointing: bool = False  # Enable for large models to save memory

@dataclass
class TokenizerConfig:
    """Enhanced configuration for tokenizer."""
    vocab_size: int = 50257
    special_tokens: Dict[str, int] = field(default_factory=lambda: {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3,
        '<mask>': 4  # Added for potential fine-tuning tasks
    })
    min_frequency: int = 2  # Minimum frequency for a token to be included
    max_token_length: int = 50  # Maximum length of a single token

@dataclass
class TrainingConfig:
    """Enhanced configuration for training."""
    # Basic training parameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 8
    max_epochs: int = 10
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5  # For cosine scheduling
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Dataset parameters
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    max_seq_length: int = 1024
    
    # Optimization
    optimizer: str = "adamw"  # choices: ["adam", "adamw", "adafactor"]
    scheduler: str = "cosine"  # choices: ["linear", "cosine", "constant"]
    gradient_checkpointing: bool = False
    fp16_training: bool = False  # Enable mixed precision training
    
    # Logging and saving
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 5  # Maximum number of checkpoints to keep
    
    # Visualization
    plot_training_progress: bool = True
    wandb_project: Optional[str] = "gpt2-training"
    wandb_entity: Optional[str] = None
    
    # Hardware
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu: int = field(default_factory=lambda: torch.cuda.device_count())
    
    # Distributed training
    local_rank: int = -1
    distributed_training: bool = False

@dataclass
class InferenceConfig:
    """Enhanced configuration for inference."""
    # Generation parameters
    max_length: int = 100
    min_length: int = 0
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    num_return_sequences: int = 1
    num_beams: int = 1  # For beam search
    early_stopping: bool = True
    
    # Output formatting
    remove_special_tokens: bool = True
    clean_up_tokenization_spaces: bool = True
    
    # Hardware
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 4
    
    # Caching
    use_cache: bool = True
    cache_dir: Optional[Path] = None

class ProjectConfig:
    """Enhanced main configuration class combining all configs."""
    
    def __init__(
        self,
        paths: Optional[PathConfig] = None,
        model: Optional[ModelConfig] = None,
        tokenizer: Optional[TokenizerConfig] = None,
        training: Optional[TrainingConfig] = None,
        inference: Optional[InferenceConfig] = None
    ):
        self._paths = paths or PathConfig()
        self._model = model or ModelConfig()
        self._tokenizer = tokenizer or TokenizerConfig()
        self._training = training or TrainingConfig()
        self._inference = inference or InferenceConfig()
        
        # Validate configurations
        self._validate_configs()
    
    @property
    def paths(self) -> PathConfig:
        """Get paths configuration."""
        return self._paths
    
    @property
    def model(self) -> ModelConfig:
        """Get model configuration."""
        return self._model
    
    @property
    def tokenizer(self) -> TokenizerConfig:
        """Get tokenizer configuration."""
        return self._tokenizer
    
    @property
    def training(self) -> TrainingConfig:
        """Get training configuration."""
        return self._training
    
    @property
    def inference(self) -> InferenceConfig:
        """Get inference configuration."""
        return self._inference
    
    def _validate_configs(self):
        """Validate configuration parameters."""
        try:
            # Validate model configuration
            assert self.model.hidden_size % self.model.num_heads == 0, \
                f"Hidden size ({self.model.hidden_size}) must be divisible by number of attention heads ({self.model.num_heads})"
            assert self.model.vocab_size >= len(self.tokenizer.special_tokens), \
                "Vocabulary size must be greater than number of special tokens"
            
            # Validate training configuration
            assert abs(sum([self.training.train_size, self.training.val_size, self.training.test_size]) - 1.0) < 1e-6, \
                "Dataset split ratios must sum to 1"
            
            if self.training.fp16_training:
                assert torch.cuda.is_available(), "FP16 training requires CUDA"
        except AssertionError as e:
            raise ValueError(str(e))
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        self.save(path)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            'paths': {k: str(v) if isinstance(v, Path) else v 
                     for k, v in vars(self.paths).items()},
            'model': vars(self.model),
            'tokenizer': vars(self.tokenizer),
            'training': vars(self.training),
            'inference': vars(self.inference)
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ProjectConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert path strings back to Path objects
        paths_dict = {k: Path(v) if isinstance(v, str) else v 
                     for k, v in config_dict['paths'].items()}
        
        return cls(
            paths=PathConfig(**paths_dict),
            model=ModelConfig(**config_dict['model']),
            tokenizer=TokenizerConfig(**config_dict['tokenizer']),
            training=TrainingConfig(**config_dict['training']),
            inference=InferenceConfig(**config_dict['inference'])
        )

def get_default_config() -> ProjectConfig:
    """Get default project configuration."""
    return ProjectConfig()