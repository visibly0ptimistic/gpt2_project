#!/usr/bin/env python
import os
import sys
import torch
import logging
from pathlib import Path
import requests
import tarfile
from tqdm import tqdm
import argparse
import shutil
import json
from typing import Optional

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
    """Handles downloading and preparing the OpenWebText dataset."""
    
    OPENWEBTEXT_URL = "https://zenodo.org/record/3834942/files/openwebtext.tar.xz"
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.download_dir = self.config.paths.data_dir / "downloads"
        self.raw_dir = self.config.paths.data_dir / "raw"
        self.processed_dir = self.config.paths.data_dir / "processed"
        
        # Create directories
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, filename: str) -> Path:
        """Download file with progress bar."""
        filepath = self.download_dir / filename
        
        if filepath.exists():
            logger.info(f"File already exists: {filepath}")
            return filepath
        
        logger.info(f"Downloading {url} to {filepath}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        
        return filepath
    
    def extract_archive(self, archive_path: Path) -> None:
        """Extract tar.xz archive."""
        logger.info(f"Extracting {archive_path} to {self.raw_dir}")
        
        with tarfile.open(archive_path, 'r:xz') as tar:
            tar.extractall(path=self.raw_dir)
    
    def prepare_dataset(self) -> None:
        """Download and prepare the OpenWebText dataset."""
        try:
            # Download dataset
            filename = "openwebtext.tar.xz"
            archive_path = self.download_file(self.OPENWEBTEXT_URL, filename)
            
            # Extract archive
            self.extract_archive(archive_path)
            
            # Save dataset info
            dataset_info = {
                'name': 'OpenWebText',
                'url': self.OPENWEBTEXT_URL,
                'download_date': str(Path(archive_path).stat().st_mtime),
                'size': Path(archive_path).stat().st_size,
                'description': 'Open-source replication of the WebText dataset used to train GPT-2'
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
        shutil.rmtree(config.paths.data_dir / "downloads", ignore_errors=True)
        shutil.rmtree(config.paths.data_dir / "raw", ignore_errors=True)
        shutil.rmtree(config.paths.data_dir / "processed", ignore_errors=True)
    
    # Initialize downloader and prepare dataset
    downloader = DatasetDownloader(config)
    downloader.prepare_dataset()

if __name__ == "__main__":
    main()