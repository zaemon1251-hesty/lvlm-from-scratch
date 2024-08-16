from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"
        )
        
        self.num_patches = (self.image_size //self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1,-1)),
            persistent=False
        )
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        
        # patch_embeds = B, E, num patch H, num_patch W
        patch_embeds = self.patch_embedding(pixel_values)
        
        # embeddings = B, E, Num_patches
        embeddings = patch_embeds.flatten(2)
        
        # embeddings = B, Num_patches, E
        embeddings = embeddings.transpose(1, 2)
        
        # add position feature
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        return embeddings


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipAttention(nn.Module):
    """Multihead Attention from 'Attention is all you need'"""
    
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        
        # splitting embed vector into several heads makes some profit e.g.) 
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
    
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # query_states = Batch_size, Num_heads, Num_patches(seq_len), Head_dim
        # transpose() replace seq_len and self.num_heads each other
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # formula Q * K^T / sqrt(d_k)
        # attn_weights = Batch_size, Num_heads, Num_patches, Num_patches
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3))*self.scale)
        
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(f"Invalid Attention size: {attn_weights.size()=}")
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # apply only training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # attn_output = Batch_size, Num_heads, Num_patches, Head_dim
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(f"Invalid Attention Output size: {attn_output.size()=}")
        
        # Batch_size, Num_patches, Num_heads, Head_dim
        attn_output = attn_output.transpose(1,2).contiguous()
        
        # attn_output = batch_size, Num_patches, Embed_dim
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        # linear transform (aware of results from each attention heads)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor):
        # hidden_states = B, Num_patches, Embed_dim
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = hidden_states + residual
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        
        last_hidden_state = self.encoder(hidden_states)
        
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        return last_hidden_state

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values):
        # (B, C, H, W) -> (B, Patch_num, Embed_dim)
        return self.vision_model(pixel_values=pixel_values)

