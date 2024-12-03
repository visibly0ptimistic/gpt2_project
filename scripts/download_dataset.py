#!/usr/bin/env python
import os
import sys
import torch
import logging
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm
import argparse
import shutil
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import ProjectConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Handles downloading and preparing datasets."""
    
    TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.raw_dir = self.config.paths.data_dir / "raw"
        self.processed_dir = self.config.paths.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self) -> None:
        """Download the dataset file."""
        filepath = self.raw_dir / "shakespeare.txt"
        
        if filepath.exists():
            logger.info(f"File already exists: {filepath}")
            return filepath
        
        logger.info(f"Downloading {self.TINY_SHAKESPEARE_URL}")
        response = requests.get(self.TINY_SHAKESPEARE_URL)
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded to {filepath}")
        return filepath
    
    def prepare_dataset(self) -> None:
        """Download and prepare the dataset."""
        try:
            # Download dataset
            self.download_file()
            
            # Save dataset info
            dataset_info = {
                'name': 'Tiny Shakespeare',
                'url': self.TINY_SHAKESPEARE_URL,
                'description': 'A small Shakespeare dataset for testing'
            }
            
            with open(self.processed_dir / 'dataset_info.json', 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            logger.info("Dataset preparation completed!")
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {str(e)}")
            raise

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing dataset files before downloading"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = ProjectConfig.load(args.config)
    else:
        config = ProjectConfig()
    
    # Clean if requested
    if args.clean:
        logger.info("Cleaning existing dataset files...")
        shutil.rmtree(config.paths.data_dir / "raw", ignore_errors=True)
        shutil.rmtree(config.paths.data_dir / "processed", ignore_errors=True)
    
    # Initialize downloader and prepare dataset
    downloader = DatasetDownloader(config)
    downloader.prepare_dataset()

if __name__ == "__main__":
    main()