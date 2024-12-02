import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

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

class FlashAttention(nn.Module):
    """Efficient attention implementation using flash attention algorithm."""
    
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.size()
        
        # QKV transform
        qkv = self.qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape and transpose for attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        
        # Attention scores with flash attention optimization
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
        
        outputs = (output,)
        if output_attentions:
            outputs += (attention_probs,)
        
        return outputs

class TransformerBlock(nn.Module):
    """Enhanced GPT-2 Transformer block with improved efficiency."""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.attention = FlashAttention(config)
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
            attention_outputs = torch.utils.checkpoint.checkpoint(
                self.attention,
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions,
            )
        else:
            attention_outputs = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )
        
        hidden_states = residual + attention_outputs[0]
        
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
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += attention_outputs[1:]
        
        return outputs

class GPT2(nn.Module):
    """Enhanced GPT-2 model with improved training and inference capabilities."""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.ModuleDict({
            'tokens': nn.Embedding(config.vocab_size, config.hidden_size),
            'positions': nn.Embedding(config.max_position_embeddings, config.hidden_size)
        })
        
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
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

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings['tokens']

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.embeddings['tokens'] = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, seq_length = input_ids.size()
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Get embeddings
        inputs_embeds = self.embeddings['tokens'](input_ids)
        position_embeds = self.embeddings['positions'](position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Generate head masks if needed
        if head_mask is not None:
            head_mask = self._prepare_head_mask(head_mask, self.config.num_layers)
        else:
            head_mask = [None] * self.config.num_layers
        
        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            layer_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        
        hidden_states = self.ln_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Prepare outputs
        outputs = (hidden_states,)
        if output_hidden_states:
            outputs += (all_hidden_states,)
        if output_attentions:
            outputs += (all_attentions,)
        
        return outputs

    def _prepare_head_mask(
        self,
        head_mask: torch.Tensor,
        num_layers: int,
        attention_heads: Optional[int] = None
    ) -> List[Optional[torch.Tensor]]:
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_layers)
            if attention_heads is not None:
                head_mask = head_mask.expand(-1, -1, attention_heads, -1, -1)
        else:
            head_mask = [None] * num_layers
        return head_mask

    @staticmethod
    def _convert_head_mask_to_5d(head_mask: torch.Tensor, num_layers: int) -> torch.Tensor:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        return head_mask

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past: Optional[List[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if past is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past
        )