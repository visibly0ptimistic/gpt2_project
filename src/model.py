import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
from pathlib import Path

@dataclass
class GPT2Config:
    """Configuration class for GPT-2 model parameters."""
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

class NewGELU(nn.Module):
    """Implementation of the GELU activation function.
    Faster and more accurate than nn.GELU."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class MultiHeadAttention(nn.Module):
    """Efficient attention implementation using MultiHeadAttention algorithm."""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_size = config.hidden_size // config.num_heads
        self.dropout = config.dropout
        self.scale = self.head_size ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.size()
        
        # QKV transform
        qkv = self.qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape and transpose for attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        
        # Attention scores with MultiHeadAttention optimization
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.hidden_size)
        
        # Output projection
        output = self.out_proj(context)
        output = self.resid_dropout(output)
        
        return output

class TransformerBlock(nn.Module):
    """Enhanced GPT-2 Transformer block with improved efficiency."""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            NewGELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )
        
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        
        # Attention block with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            attention_output = torch.utils.checkpoint.checkpoint(
                self.attention,
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions,
            )
        else:
            attention_output = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )
        
        hidden_states = residual + attention_output
        
        # MLP block with optional gradient checkpointing
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        
        if self.gradient_checkpointing and self.training:
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.mlp,
                hidden_states,
            )
        else:
            hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        return hidden_states

class GPT2(nn.Module):
    """Enhanced GPT-2 model with improved training and inference capabilities."""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights between input embeddings and output layer
        self.head.weight = self.token_embeddings.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get position IDs for input sequence."""
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        return position_ids.unsqueeze(0).expand_as(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = input_ids.size()
        
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Get embeddings
        inputs_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        # Process through transformer blocks
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )
        
        hidden_states = self.ln_f(hidden_states)
        
        # Get logits
        logits = self.head(hidden_states)
        
        return logits, hidden_states

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate text continuation."""
        self.eval()
        batch_size = input_ids.size(0)
        cur_len = input_ids.size(1)
        
        with torch.no_grad():
            while cur_len < max_length:
                # Forward pass
                outputs = self.forward(input_ids)
                logits = outputs[0]
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append token
                input_ids = torch.cat((input_ids, next_token), dim=1)
                cur_len = input_ids.size(1)
                
        return input_ids

    def save_pretrained(self, save_dir: str) -> None:
        """Save model and configuration."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = save_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=4)
        
        # Save model weights
        model_path = save_dir / 'pytorch_model.bin'
        torch.save(self.state_dict(), model_path)

    @classmethod
    def from_pretrained(cls, model_dir: str) -> 'GPT2':
        """Load model from pretrained directory."""
        model_dir = Path(model_dir)
        
        # Load configuration
        config_path = model_dir / 'config.json'
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = GPT2Config(**config_dict)
        
        # Initialize model
        model = cls(config)
        
        # Load weights
        model_path = model_dir / 'pytorch_model.bin'
        model.load_state_dict(torch.load(model_path))
        
        return model