import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import wandb
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
from typing import Optional, Tuple, Dict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model import GPT2, GPT2Config
from src.config import ProjectConfig
from src.tokenizer import BPETokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Custom dataset for training GPT-2."""
    
    def __init__(
        self,
        data_path: Path,
        max_length: int,
        tokenizer: BPETokenizer
    ):
        self.data = torch.load(data_path)
        self.max_length = max_length
        self.tokenizer = tokenizer
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # Get sequence and ensure it's properly sized
        sequence = self.data[idx]
        if len(sequence) > self.max_length:
            # Random crop for training
            start_idx = torch.randint(0, len(sequence) - self.max_length, (1,))
            sequence = sequence[start_idx:start_idx + self.max_length]
        else:
            # Pad if necessary
            padding = [self.tokenizer.special_tokens['<pad>']] * (self.max_length - len(sequence))
            sequence = sequence + padding
        
        return torch.tensor(sequence, dtype=torch.long)

class Trainer:
    """Enhanced GPT-2 trainer with modern training features."""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.device = torch.device(config.training.device)
        
        # Setup model
        logger.info("Initializing model...")
        self.model = self._setup_model()
        
        # Setup tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = self._setup_tokenizer()
        
        # Setup data
        logger.info("Setting up data loaders...")
        self.train_loader, self.val_loader = self._setup_data()
        
        # Setup training components
        self.optimizer = self._setup_optimizer()
        self.scaler = GradScaler() if config.training.fp16_training else None
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Initialize tracking metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

    def _setup_model(self) -> GPT2:
        """Initialize and prepare model."""
        model = GPT2(self.config.model)
        
        if self.config.training.distributed_training:
            model = nn.parallel.DistributedDataParallel(model)
        elif torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        model = model.to(self.device)
        
        if self.config.training.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model

    def _setup_tokenizer(self) -> BPETokenizer:
        """Load tokenizer."""
        tokenizer = BPETokenizer(vocab_size=self.config.model.vocab_size)
        tokenizer.load(self.config.paths.tokenizer_dir)
        return tokenizer

    def _setup_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders."""
        # Create datasets
        train_dataset = TextDataset(
            self.config.paths.train_dataset_path,
            self.config.training.max_seq_length,
            self.tokenizer
        )
        
        val_dataset = TextDataset(
            self.config.paths.val_dataset_path,
            self.config.training.max_seq_length,
            self.tokenizer
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer with learning rate schedule."""
        # Prepare optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.training.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer

    def _convert_paths_to_str(self, obj):
        """Convert all Path objects to strings in a config object."""
        if isinstance(obj, dict):
            return {k: self._convert_paths_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_paths_to_str(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Initialize wandb if enabled
        if self.config.training.use_wandb:
            # Convert config to JSON serializable format
            config_dict = vars(self.config)
            wandb_config = self._convert_paths_to_str(config_dict)
            
            wandb.init(
                project=self.config.training.wandb_project,
                entity=self.config.training.wandb_entity,
                config=wandb_config
            )
        
        try:
            for epoch in range(self.config.training.max_epochs):
                # Training phase
                train_loss = self._train_epoch(epoch)
                
                # Validation phase
                val_loss = self._validate_epoch(epoch)
                
                # Log metrics
                self._log_metrics(epoch, train_loss, val_loss)
                
                # Save checkpoint if best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint('best_model.pt', epoch, val_loss)
                
                # Regular checkpoint saving
                if epoch % self.config.training.save_steps == 0:
                    self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch, val_loss)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            # Save final model
            self._save_checkpoint('final_model.pt', epoch, val_loss)
            
            if self.config.training.use_wandb:
                wandb.finish()

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.training.max_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                loss = self._training_step(batch)
                total_loss += loss
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.4f}'})
                
                # Log step metrics
                if self.global_step % self.config.training.logging_steps == 0:
                    self._log_step_metrics(loss)
                
                self.global_step += 1
        
        return total_loss / len(self.train_loader)

    def _training_step(self, batch: torch.Tensor) -> float:
        """Perform one training step."""
        batch = batch.to(self.device)
        
        # Forward pass
        with autocast(enabled=self.config.training.fp16_training):
            logits, _ = self.model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch[:, 1:].contiguous().view(-1)
            )
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.training.gradient_accumulation_steps
        
        # Backward pass
        if self.config.training.fp16_training:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights if gradient accumulation is complete
        if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
            if self.config.training.fp16_training:
                self.scaler.unscale_(self.optimizer)
                
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            
            # Update weights
            if self.config.training.fp16_training:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        return loss.item()

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            batch = batch.to(self.device)
            logits, _ = self.model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch[:, 1:].contiguous().view(-1)
            )
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader)

    def _log_metrics(self, epoch: int, train_loss: float, val_loss: float):
        """Log training metrics."""
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # Update tracking metrics
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['learning_rates'].append(metrics['learning_rate'])
        
        # Log to wandb if enabled
        if self.config.training.use_wandb:
            wandb.log(metrics, step=self.global_step)
        
        # Log to console
        logger.info(
            f"Epoch {epoch+1}/{self.config.training.max_epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f}"
        )

    def _log_step_metrics(self, loss: float):
        """Log metrics for training step."""
        if self.config.training.use_wandb:
            wandb.log({
                'step_loss': loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }, step=self.global_step)

    def _save_checkpoint(self, filename: str, epoch: int, loss: float):
        """Save training checkpoint."""
        checkpoint_path = self.config.paths.checkpoints_dir / filename
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config.__dict__,
            'metrics': self.metrics
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state
        self.global_step = checkpoint['global_step']
        self.metrics = checkpoint['metrics']
        
        return checkpoint['epoch']

def main():
    """Main training function."""
    # Load configuration
    config = ProjectConfig()
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()