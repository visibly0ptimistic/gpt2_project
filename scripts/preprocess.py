import os
import sys
import torch
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import multiprocessing as mp
from itertools import chain
import json
import random
import argparse
from datetime import datetime
import shutil

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.tokenizer import BPETokenizer
from src.config import ProjectConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Enhanced data preprocessing pipeline for GPT-2 training."""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.tokenizer = None
        
        # Create necessary directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create required directories."""
        directories = [
            self.config.paths.data_dir,
            self.config.paths.processed_dir,
            self.config.paths.tokenizer_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self, data_path: Path) -> List[str]:
        """
        Load and clean raw text data.
        
        Args:
            data_path: Path to raw data file or directory
            
        Returns:
            List of cleaned text samples
        """
        logger.info(f"Loading data from {data_path}")
        texts = []
        
        if data_path.is_file():
            # Single file processing
            texts.extend(self._process_file(data_path))
        else:
            # Directory processing
            for file_path in tqdm(list(data_path.rglob('*.txt')), desc="Loading files"):
                texts.extend(self._process_file(file_path))
        
        logger.info(f"Loaded {len(texts)} text samples")
        return texts
    
    def _process_file(self, file_path: Path) -> List[str]:
        """Process a single text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split into samples and clean
            samples = self._clean_text(text)
            
            return samples
            
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {str(e)}")
            return []
    
    def _clean_text(self, text: str) -> List[str]:
        """Clean and split text into samples."""
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        # Clean each paragraph
        cleaned = []
        for para in paragraphs:
            # Basic cleaning
            para = para.strip()
            
            # Skip if too short
            if len(para.split()) < 10:
                continue
            
            # Apply cleaning rules
            para = self._apply_cleaning_rules(para)
            
            if para:
                cleaned.append(para)
        
        return cleaned
    
    def _apply_cleaning_rules(self, text: str) -> str:
        """Apply cleaning rules to text."""
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Remove special characters
        text = text.replace('\t', ' ')
        text = text.replace('\r', ' ')
        
        # Normalize quotes
        text = text.replace('"', '"')
        text = text.replace('"', '"')
        text = text.replace(''', "'")
        text = text.replace(''', "'")
        
        return text.strip()
    
    def train_tokenizer(self, texts: List[str]) -> None:
        """Train the BPE tokenizer on the text data."""
        logger.info("Training tokenizer...")
        
        # Initialize tokenizer
        self.tokenizer = BPETokenizer(
            vocab_size=self.config.model.vocab_size,
            min_frequency=2
        )
        
        # Train tokenizer
        self.tokenizer.train(texts, verbose=True)
        
        # Save tokenizer
        self.tokenizer.save(self.config.paths.tokenizer_dir)
        logger.info(f"Saved tokenizer to {self.config.paths.tokenizer_dir}")
    
    def _encode_chunk(self, texts: List[str]) -> List[List[int]]:
        """Encode a chunk of texts using multiprocessing."""
        return [
            self.tokenizer.encode(
                text,
                max_length=self.config.training.max_seq_length,
                padding=True,
                truncation=True
            )
            for text in texts
        ]
    
    def encode_texts(self, texts: List[str]) -> List[List[int]]:
        """Encode texts using multiprocessing."""
        logger.info("Encoding texts...")
        
        # Split texts into chunks for multiprocessing
        num_cpus = mp.cpu_count()
        chunk_size = len(texts) // num_cpus + 1
        text_chunks = [
            texts[i:i + chunk_size]
            for i in range(0, len(texts), chunk_size)
        ]
        
        # Encode chunks in parallel
        with mp.Pool(num_cpus) as pool:
            encoded_chunks = list(tqdm(
                pool.imap(self._encode_chunk, text_chunks),
                total=len(text_chunks),
                desc="Encoding chunks"
            ))
        
        # Flatten results
        encoded_texts = list(chain.from_iterable(encoded_chunks))
        
        return encoded_texts
    
    def create_data_splits(
        self,
        encoded_texts: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create train/validation/test splits."""
        logger.info("Creating data splits...")
        
        # Shuffle data
        random.shuffle(encoded_texts)
        
        # Calculate split sizes
        total_size = len(encoded_texts)
        train_size = int(total_size * self.config.training.train_size)
        val_size = int(total_size * self.config.training.val_size)
        
        # Split data
        train_data = encoded_texts[:train_size]
        val_data = encoded_texts[train_size:train_size + val_size]
        test_data = encoded_texts[train_size + val_size:]
        
        # Convert to tensors
        train_tensor = torch.tensor(train_data, dtype=torch.long)
        val_tensor = torch.tensor(val_data, dtype=torch.long)
        test_tensor = torch.tensor(test_data, dtype=torch.long)
        
        logger.info(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_tensor, val_tensor, test_tensor
    
    def save_splits(
        self,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        test_data: torch.Tensor
    ) -> None:
        """Save data splits to disk."""
        logger.info("Saving data splits...")
        
        # Save tensors
        torch.save(train_data, self.config.paths.train_dataset_path)
        torch.save(val_data, self.config.paths.val_dataset_path)
        torch.save(test_data, self.config.paths.test_dataset_path)
        
        # Save split info
        split_info = {
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'max_seq_length': self.config.training.max_seq_length,
            'vocab_size': self.config.model.vocab_size,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.config.paths.processed_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info("Data splits saved successfully")
    
    def preprocess_data(self) -> None:
        """Run the complete preprocessing pipeline."""
        try:
            # Load raw data
            texts = self.load_raw_data(self.config.paths.data_dir)
            
            # Train tokenizer
            self.train_tokenizer(texts)
            
            # Encode texts
            encoded_texts = self.encode_texts(texts)
            
            # Create data splits
            train_data, val_data, test_data = self.create_data_splits(encoded_texts)
            
            # Save splits
            self.save_splits(train_data, val_data, test_data)
            
            logger.info("Preprocessing completed successfully!")
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess data for GPT-2 training")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to raw data directory"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = ProjectConfig.load(args.config)
    else:
        config = ProjectConfig()
    
    # Override data directory if specified
    if args.data_dir:
        config.paths.data_dir = Path(args.data_dir)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Run preprocessing
    preprocessor.preprocess_data()

if __name__ == "__main__":
    main()