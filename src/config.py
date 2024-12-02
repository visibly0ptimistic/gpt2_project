from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch
import json

class PathEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling Path objects."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

@dataclass
class PathConfig:
    """Enhanced configuration for project paths."""
    project_root: Path
    data_dir: Optional[Path] = None
    checkpoints_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None
    datasets: Optional[Dict[str, Path]] = None
    processed_dir: Optional[Path] = None
    train_dataset_path: Optional[Path] = None
    val_dataset_path: Optional[Path] = None
    test_dataset_path: Optional[Path] = None
    tokenizer_dir: Optional[Path] = None
    vocab_file: Optional[Path] = None
    merges_file: Optional[Path] = None

    def __post_init__(self):
        """Initialize and create all necessary directories."""
        # Set default paths relative to project_root if not provided
        self.data_dir = self.data_dir or self.project_root / "data"
        self.checkpoints_dir = self.checkpoints_dir or self.project_root / "checkpoints"
        self.logs_dir = self.logs_dir or self.project_root / "logs"
        
        # Set dataset paths
        self.datasets = self.datasets or {
            'openwebtext': self.data_dir / "openwebtext",
            'books': self.data_dir / "books",
            'custom': self.data_dir / "custom"
        }
        
        # Set processed data paths
        self.processed_dir = self.processed_dir or self.data_dir / "processed"
        self.train_dataset_path = self.train_dataset_path or self.processed_dir / "train.pt"
        self.val_dataset_path = self.val_dataset_path or self.processed_dir / "val.pt"
        self.test_dataset_path = self.test_dataset_path or self.processed_dir / "test.pt"
        
        # Set tokenizer paths
        self.tokenizer_dir = self.tokenizer_dir or self.data_dir / "tokenizer"
        self.vocab_file = self.vocab_file or self.tokenizer_dir / "vocab.json"
        self.merges_file = self.merges_file or self.tokenizer_dir / "merges.txt"

@dataclass
class ModelConfig:
    """Enhanced configuration for model architecture."""
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    use_cache: bool = True
    gradient_checkpointing: bool = False

@dataclass
class TokenizerConfig:
    """Enhanced configuration for tokenizer."""
    vocab_size: int = 50257
    special_tokens: Dict[str, int] = field(default_factory=lambda: {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3,
        '<mask>': 4
    })
    min_frequency: int = 2
    max_token_length: int = 50

@dataclass
class TrainingConfig:
    """Enhanced configuration for training."""
    batch_size: int = 32
    gradient_accumulation_steps: int = 8
    max_epochs: int = 10
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    max_seq_length: int = 1024
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_checkpointing: bool = False
    fp16_training: bool = False
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 5
    plot_training_progress: bool = True
    wandb_project: Optional[str] = "gpt2-training"
    wandb_entity: Optional[str] = None
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu: int = field(default_factory=lambda: torch.cuda.device_count())
    local_rank: int = -1
    distributed_training: bool = False

@dataclass
class InferenceConfig:
    """Enhanced configuration for inference."""
    max_length: int = 100
    min_length: int = 0
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    num_return_sequences: int = 1
    num_beams: int = 1
    early_stopping: bool = True
    remove_special_tokens: bool = True
    clean_up_tokenization_spaces: bool = True
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 4
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
        # Initialize with default project root if paths not provided
        if paths is None:
            paths = PathConfig(project_root=Path.cwd())
            
        # Initialize configs
        self._paths = paths
        self._model = model or ModelConfig()
        self._tokenizer = tokenizer or TokenizerConfig()
        self._training = training or TrainingConfig()
        self._inference = inference or InferenceConfig()
        
        # Create necessary directories
        self._ensure_directories_exist()
        
        # Validate configurations
        self._validate_configs()
    
    def _ensure_directories_exist(self) -> None:
        """Ensure all required directories exist."""
        try:
            # Create base directories first
            self._paths.data_dir.mkdir(parents=True, exist_ok=True)
            self._paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self._paths.logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Then create subdirectories
            self._paths.processed_dir.mkdir(parents=True, exist_ok=True)
            self._paths.tokenizer_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dataset directories
            for dataset_dir in self._paths.datasets.values():
                dataset_dir.mkdir(parents=True, exist_ok=True)
                
        except Exception as e:
            raise RuntimeError(f"Failed to create directory structure: {str(e)}")

    @property
    def paths(self) -> PathConfig:
        return self._paths

    @paths.setter
    def paths(self, value: PathConfig) -> None:
        raise AttributeError("Cannot modify paths after initialization")

    @property
    def model(self) -> ModelConfig:
        return self._model

    @model.setter
    def model(self, value: ModelConfig) -> None:
        self._model = value
        self._validate_configs()

    @property
    def tokenizer(self) -> TokenizerConfig:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: TokenizerConfig) -> None:
        self._tokenizer = value
        self._validate_configs()

    @property
    def training(self) -> TrainingConfig:
        return self._training

    @training.setter
    def training(self, value: TrainingConfig) -> None:
        self._training = value
        self._validate_configs()

    @property
    def inference(self) -> InferenceConfig:
        return self._inference

    @inference.setter
    def inference(self, value: InferenceConfig) -> None:
        self._inference = value

    def _validate_configs(self) -> None:
        """Validate configuration parameters."""
        try:
            # Validate model configuration
            assert self.model.hidden_size % self.model.num_heads == 0, \
                f"Hidden size ({self.model.hidden_size}) must be divisible by number of attention heads ({self.model.num_heads})"
            assert self.model.vocab_size >= len(self.tokenizer.special_tokens), \
                "Vocabulary size must be greater than number of special tokens"
            
            # Validate training configuration
            split_sum = self.training.train_size + self.training.val_size + self.training.test_size
            assert abs(split_sum - 1.0) < 1e-6, \
                f"Dataset split ratios must sum to 1, got {split_sum}"
            
            if self.training.fp16_training:
                assert torch.cuda.is_available(), "FP16 training requires CUDA"
                
        except AssertionError as e:
            raise ValueError(str(e))

    def to_json(self, path: str) -> None:
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
            json.dump(config_dict, f, indent=2, cls=PathEncoder)

    @classmethod
    def from_json(cls, path: str) -> 'ProjectConfig':
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

    @classmethod
    def load(cls, path: str) -> 'ProjectConfig':
        """Alias for from_json for backward compatibility."""
        return cls.from_json(path)

def get_default_config() -> ProjectConfig:
    """Get default project configuration."""
    return ProjectConfig()