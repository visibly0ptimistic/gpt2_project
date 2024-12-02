import os
import sys
import torch
import logging
from pathlib import Path
from typing import List, Optional, Dict
import time
import json
import numpy as np
from tqdm import tqdm
import argparse
import readline  # Enables arrow key navigation in input
import curses
from termcolor import colored

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model import GPT2
from src.tokenizer import BPETokenizer
from src.config import ProjectConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InteractiveDemo:
    """Interactive demo for GPT-2 text generation."""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.device = torch.device(config.inference.device)
        
        # Load model and tokenizer
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        
        # Generation settings
        self.default_settings = {
            'max_length': config.inference.max_length,
            'temperature': config.inference.temperature,
            'top_k': config.inference.top_k,
            'top_p': config.inference.top_p,
            'num_return_sequences': config.inference.num_return_sequences
        }
        
        # Generation history
        self.history: List[Dict] = []
    
    def _load_model(self) -> GPT2:
        """Load the trained model."""
        logger.info("Loading model...")
        model = GPT2.from_pretrained(self.config.paths.best_model_path)
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_tokenizer(self) -> BPETokenizer:
        """Load the tokenizer."""
        logger.info("Loading tokenizer...")
        tokenizer = BPETokenizer()
        tokenizer.load(self.config.paths.tokenizer_dir)
        return tokenizer
    
    def generate_text(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: Optional[int] = None,
        show_progress: bool = True
    ) -> List[str]:
        """Generate text based on prompt with visual progress."""
        # Use default settings if not specified
        max_length = max_length or self.default_settings['max_length']
        temperature = temperature or self.default_settings['temperature']
        top_k = top_k or self.default_settings['top_k']
        top_p = top_p or self.default_settings['top_p']
        num_return_sequences = num_return_sequences or self.default_settings['num_return_sequences']
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids).unsqueeze(0).repeat(num_return_sequences, 1)
        input_tensor = input_tensor.to(self.device)
        
        # Generate text with progress bar
        sequences = []
        with torch.no_grad():
            if show_progress:
                pbar = tqdm(total=max_length, desc="Generating")
            
            while input_tensor.shape[1] < max_length:
                # Get next token predictions
                outputs = self.model(input_tensor)
                next_token_logits = outputs[0][:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next tokens
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                
                # Update input tensor
                input_tensor = torch.cat([input_tensor, next_tokens], dim=1)
                
                if show_progress:
                    pbar.update(1)
                
                # Check if any sequence is complete
                if self.tokenizer.special_tokens['<eos>'] in next_tokens:
                    break
            
            if show_progress:
                pbar.close()
        
        # Decode generated sequences
        for sequence in input_tensor:
            text = self.tokenizer.decode(sequence.tolist())
            sequences.append(text)
        
        # Save to history
        self.history.append({
            'prompt': prompt,
            'settings': {
                'max_length': max_length,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p
            },
            'generations': sequences
        })
        
        return sequences

    def _print_settings(self) -> None:
        """Print current generation settings."""
        print("\nCurrent Settings:")
        print(f"  Max Length: {self.default_settings['max_length']}")
        print(f"  Temperature: {self.default_settings['temperature']}")
        print(f"  Top-k: {self.default_settings['top_k']}")
        print(f"  Top-p: {self.default_settings['top_p']}")
        print(f"  Num Sequences: {self.default_settings['num_return_sequences']}")

    def _update_settings(self) -> None:
        """Update generation settings interactively."""
        print("\nEnter new values (press Enter to keep current value):")
        
        for key in self.default_settings:
            current = self.default_settings[key]
            new_value = input(f"{key} ({current}): ").strip()
            
            if new_value:
                try:
                    if isinstance(current, int):
                        self.default_settings[key] = int(new_value)
                    else:
                        self.default_settings[key] = float(new_value)
                except ValueError:
                    print(f"Invalid value for {key}, keeping current value")

    def _save_history(self, filepath: str) -> None:
        """Save generation history to file."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\nHistory saved to {filepath}")

    def _load_history(self, filepath: str) -> None:
        """Load generation history from file."""
        with open(filepath, 'r') as f:
            self.history = json.load(f)
        print(f"\nLoaded {len(self.history)} historical generations")

    def _print_help(self) -> None:
        """Print help message."""
        print("\nAvailable Commands:")
        print("  /help - Show this help message")
        print("  /settings - Show current generation settings")
        print("  /update - Update generation settings")
        print("  /save <filename> - Save generation history")
        print("  /load <filename> - Load generation history")
        print("  /history - Show generation history")
        print("  /quit - Exit the demo")
        print("\nOr enter any text prompt to generate completions.")

    def run_interactive(self):
        """Run interactive demo session."""
        print("\nGPT-2 Interactive Demo")
        print("Enter '/help' for available commands")
        
        while True:
            try:
                # Get input
                prompt = input("\nPrompt: ").strip()
                
                # Handle commands
                if prompt.startswith('/'):
                    self._handle_command(prompt)
                    continue
                
                # Generate text
                if prompt:
                    generations = self.generate_text(prompt)
                    
                    # Print generations
                    print("\nGenerated texts:")
                    for i, text in enumerate(generations, 1):
                        print(f"\n{i}. {colored(text, 'green')}")
            
            except KeyboardInterrupt:
                print("\nGeneration interrupted")
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                print("An error occurred. Please try again.")
    
    def _handle_command(self, command: str) -> None:
        """Handle demo commands."""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/help':
            self._print_help()
        
        elif cmd == '/settings':
            self._print_settings()
        
        elif cmd == '/update':
            self._update_settings()
        
        elif cmd == '/save' and len(parts) > 1:
            self._save_history(parts[1])
        
        elif cmd == '/load' and len(parts) > 1:
            self._load_history(parts[1])
        
        elif cmd == '/history':
            print("\nGeneration History:")
            for i, entry in enumerate(self.history, 1):
                print(f"\n{i}. Prompt: {entry['prompt']}")
                print(f"   Settings: {entry['settings']}")
                print(f"   Generations: {len(entry['generations'])}")
        
        elif cmd == '/quit':
            print("\nExiting demo...")
            sys.exit(0)
        
        else:
            print("Unknown command. Type /help for available commands.")

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Interactive GPT-2 Demo")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model checkpoint"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = ProjectConfig.load(args.config)
    else:
        config = ProjectConfig()
    
    # Override model path if specified
    if args.model_path:
        config.paths.best_model_path = Path(args.model_path)
    
    # Initialize and run demo
    demo = InteractiveDemo(config)
    demo.run_interactive()

if __name__ == "__main__":
    main()