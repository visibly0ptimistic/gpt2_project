import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import pandas as pd
from collections import Counter
from datetime import datetime
import argparse
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model import GPT2
from src.tokenizer import BPETokenizer
from src.config import ProjectConfig

class ModelAnalyzer:
    """Analysis and visualization tools for GPT-2 model."""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.device = torch.device(config.inference.device)
        
        # Set style for plots
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_training_metrics(self, checkpoint_path: Path) -> None:
        """Plot training metrics from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        metrics = checkpoint['metrics']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot loss curves
        epochs = range(1, len(metrics['train_loss']) + 1)
        ax1.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss During Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot learning rate
        ax2.plot(epochs, metrics['learning_rates'], 'g-')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True)
        
        plt.tight_layout()
        save_path = self.config.paths.logs_dir / 'training_metrics.png'
        plt.savefig(save_path)
        plt.close()
        
        print(f"Training metrics plot saved to {save_path}")
    
    def analyze_attention_patterns(self, model: GPT2, text: str) -> None:
        """Visualize attention patterns for given text."""
        tokenizer = BPETokenizer()
        tokenizer.load(self.config.paths.tokenizer_dir)
        
        # Tokenize input
        input_ids = tokenizer.encode(text)
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        
        # Get attention weights
        with torch.no_grad():
            _, attention_weights = model(input_tensor, output_attentions=True)
        
        # Average attention weights across heads
        avg_attention = torch.mean(attention_weights[-1], dim=1).squeeze()
        
        # Create attention heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            avg_attention.cpu().numpy(),
            xticklabels=[tokenizer.decode([id]) for id in input_ids],
            yticklabels=[tokenizer.decode([id]) for id in input_ids],
            cmap='viridis'
        )
        plt.title('Attention Pattern Analysis')
        
        save_path = self.config.paths.logs_dir / 'attention_pattern.png'
        plt.savefig(save_path)
        plt.close()
        
        print(f"Attention pattern plot saved to {save_path}")
    
    def plot_token_distributions(self, generated_texts: List[str]) -> None:
        """Analyze token distributions in generated text."""
        tokenizer = BPETokenizer()
        tokenizer.load(self.config.paths.tokenizer_dir)
        
        # Collect token statistics
        token_counts = Counter()
        sequence_lengths = []
        
        for text in generated_texts:
            tokens = tokenizer.encode(text)
            token_counts.update(tokens)
            sequence_lengths.append(len(tokens))
        
        # Plot sequence length distribution
        plt.figure(figsize=(10, 6))
        plt.hist(sequence_lengths, bins=30)
        plt.title('Distribution of Generated Sequence Lengths')
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency')
        
        save_path = self.config.paths.logs_dir / 'sequence_lengths.png'
        plt.savefig(save_path)
        plt.close()
        
        # Plot top token frequencies
        top_tokens = dict(token_counts.most_common(20))
        plt.figure(figsize=(12, 6))
        plt.bar(
            range(len(top_tokens)),
            list(top_tokens.values())
        )
        plt.xticks(
            range(len(top_tokens)),
            [tokenizer.decode([t]) for t in top_tokens.keys()],
            rotation=45
        )
        plt.title('Most Common Tokens in Generated Text')
        plt.xlabel('Token')
        plt.ylabel('Frequency')
        
        save_path = self.config.paths.logs_dir / 'token_frequencies.png'
        plt.savefig(save_path)
        plt.close()
        
        print(f"Token distribution plots saved to {self.config.paths.logs_dir}")
    
    def analyze_model_parameters(self, model: GPT2) -> Dict:
        """Analyze model parameters and architecture."""
        stats = {}
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        stats['total_parameters'] = total_params
        stats['trainable_parameters'] = trainable_params
        stats['frozen_parameters'] = total_params - trainable_params
        
        # Layer statistics
        layer_stats = {
            'attention': [],
            'ffn': [],
            'total': []
        }
        
        for block in model.blocks:
            # Attention parameters
            attn_params = sum(p.numel() for p in block.attention.parameters())
            layer_stats['attention'].append(attn_params)
            
            # FFN parameters
            ffn_params = sum(p.numel() for p in block.mlp.parameters())
            layer_stats['ffn'].append(ffn_params)
            
            # Total layer parameters
            layer_stats['total'].append(attn_params + ffn_params)
        
        stats['layer_statistics'] = layer_stats
        
        # Plot layer statistics
        plt.figure(figsize=(10, 6))
        layers = range(1, len(layer_stats['total']) + 1)
        plt.plot(layers, layer_stats['attention'], 'b-', label='Attention')
        plt.plot(layers, layer_stats['ffn'], 'r-', label='Feed-Forward')
        plt.plot(layers, layer_stats['total'], 'g-', label='Total')
        plt.title('Parameters per Layer')
        plt.xlabel('Layer')
        plt.ylabel('Number of Parameters')
        plt.legend()
        plt.grid(True)
        
        save_path = self.config.paths.logs_dir / 'layer_parameters.png'
        plt.savefig(save_path)
        plt.close()
        
        # Save statistics to file
        stats_path = self.config.paths.logs_dir / 'model_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def analyze_generation_quality(self, generations: List[Dict]) -> None:
        """Analyze quality metrics of generated text."""
        # Collect metrics
        metrics = {
            'lengths': [],
            'unique_tokens': [],
            'repetition_rate': []
        }
        
        tokenizer = BPETokenizer()
        tokenizer.load(self.config.paths.tokenizer_dir)
        
        for gen in generations:
            for text in gen['generations']:
                # Text length
                metrics['lengths'].append(len(text.split()))
                
                # Unique tokens
                tokens = tokenizer.encode(text)
                metrics['unique_tokens'].append(len(set(tokens)))
                
                # Repetition rate (ratio of repeated bigrams)
                words = text.split()
                bigrams = list(zip(words[:-1], words[1:]))
                unique_bigrams = len(set(bigrams))
                total_bigrams = len(bigrams)
                repetition_rate = 1 - (unique_bigrams / total_bigrams) if total_bigrams > 0 else 0
                metrics['repetition_rate'].append(repetition_rate)
        
        # Plot metrics
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Length distribution
        sns.histplot(metrics['lengths'], ax=ax1)
        ax1.set_title('Distribution of Generated Text Lengths')
        ax1.set_xlabel('Length (words)')
        ax1.set_ylabel('Count')
        
        # Unique tokens
        sns.histplot(metrics['unique_tokens'], ax=ax2)
        ax2.set_title('Distribution of Unique Tokens')
        ax2.set_xlabel('Number of Unique Tokens')
        ax2.set_ylabel('Count')
        
        # Repetition rate
        sns.histplot(metrics['repetition_rate'], ax=ax3)
        ax3.set_title('Distribution of Repetition Rates')
        ax3.set_xlabel('Repetition Rate')
        ax3.set_ylabel('Count')
        
        plt.tight_layout()
        save_path = self.config.paths.logs_dir / 'generation_quality.png'
        plt.savefig(save_path)
        plt.close()
        
        print(f"Generation quality analysis saved to {save_path}")
    
    def create_analysis_report(self) -> None:
        """Create comprehensive analysis report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'analysis': {}
        }
        
        # Load model
        model = GPT2.from_pretrained(self.config.paths.best_model_path)
        model = model.to(self.device)
        
        # Analyze model architecture
        report['analysis']['model_statistics'] = self.analyze_model_parameters(model)
        
        # Load and analyze generations if available
        generations_path = self.config.paths.logs_dir / 'generation_history.json'
        if generations_path.exists():
            with open(generations_path, 'r') as f:
                generations = json.load(f)
            self.analyze_generation_quality(generations)
        
        # Save report
        report_path = self.config.paths.logs_dir / 'analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Analysis report saved to {report_path}")

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="GPT-2 Model Analysis")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint for analysis"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = ProjectConfig.load(args.config)
    else:
        config = ProjectConfig()
    
    # Initialize analyzer
    analyzer = ModelAnalyzer(config)
    
    # Run analysis
    if args.checkpoint:
        analyzer.plot_training_metrics(Path(args.checkpoint))
    
    analyzer.create_analysis_report()

if __name__ == "__main__":
    main()