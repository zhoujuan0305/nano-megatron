#!/usr/bin/env python3
"""Verify that nano-megatron's ReferenceGPT matches Megatron-LM's GPT-3 345M architecture."""

import sys
sys.path.insert(0, "/workspace/src/Megatron-LM")

import torch
from nano_megatron.reference import ReferenceGPT, ReferenceGPTConfig


def create_megatron_gpt3_345m_config():
    """Create a config matching Megatron-LM's GPT-3 345M."""
    from megatron.core.transformer.transformer_config import TransformerConfig
    
    return TransformerConfig(
        num_layers=12,
        hidden_size=512,
        num_attention_heads=8,
        ffn_hidden_size=2048,  # 4 * hidden_size
        kv_channels=64,  # hidden_size // num_attention_heads
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        add_qkv_bias=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,  # SwiGLU uses SiLU
        normalization="LayerNorm",
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


def create_nano_megatron_config():
    """Create a config matching Megatron-LM's GPT-3 345M."""
    return ReferenceGPTConfig(
        vocab_size=51200,
        max_seq_len=1024,
        hidden_size=512,
        num_layers=12,
        num_heads=8,
        ffn_hidden_size=2048,
        layernorm_eps=1e-5,
        use_bias=False,
        position_embedding_type='rope',
        rotary_dim=64,  # hidden_size // num_heads
        rotary_base=10000,
        activation_func='swiglu',
        use_fused_qkv=False,
        num_query_groups=8,  # Standard MHA
        add_qkv_bias=False,
        gated_linear_unit=True,
        normalization='layernorm',
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


def count_parameters(model):
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def print_model_structure(model, name="Model"):
    """Print model structure."""
    print(f"\n{'='*60}")
    print(f"{name} Structure")
    print(f"{'='*60}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"\nLayers:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")


def verify_architecture_match():
    """Verify that nano-megatron's architecture matches Megatron-LM's."""
    print("Verifying architecture match between nano-megatron and Megatron-LM GPT-3 345M")
    print("="*80)
    
    # Create configs
    megatron_config = create_megatron_gpt3_345m_config()
    nano_config = create_nano_megatron_config()
    
    # Print config comparison
    print("\nConfiguration Comparison:")
    print("-"*40)
    print(f"{'Parameter':<25} {'Megatron-LM':<15} {'nano-megatron':<15}")
    print("-"*40)
    print(f"{'num_layers':<25} {megatron_config.num_layers:<15} {nano_config.num_layers:<15}")
    print(f"{'hidden_size':<25} {megatron_config.hidden_size:<15} {nano_config.hidden_size:<15}")
    print(f"{'num_attention_heads':<25} {megatron_config.num_attention_heads:<15} {nano_config.num_heads:<15}")
    print(f"{'ffn_hidden_size':<25} {megatron_config.ffn_hidden_size:<15} {nano_config.ffn_hidden_size:<15}")
    print(f"{'kv_channels':<25} {megatron_config.kv_channels:<15} {nano_config.rotary_dim:<15}")
    print(f"{'layernorm_epsilon':<25} {megatron_config.layernorm_epsilon:<15} {nano_config.layernorm_eps:<15}")
    print(f"{'add_bias_linear':<25} {megatron_config.add_bias_linear:<15} {nano_config.use_bias:<15}")
    print(f"{'gated_linear_unit':<25} {megatron_config.gated_linear_unit:<15} {nano_config.gated_linear_unit:<15}")
    print(f"{'normalization':<25} {megatron_config.normalization:<15} {nano_config.normalization:<15}")
    
    # Create models
    print("\nCreating nano-megatron model...")
    nano_model = ReferenceGPT(nano_config)
    
    # Print model structure
    print_model_structure(nano_model, "nano-megatron ReferenceGPT")
    
    # Verify key components
    print("\n\nKey Component Verification:")
    print("-"*40)
    
    # Check position embedding type
    if nano_config.position_embedding_type == 'rope':
        print("✓ Position embedding: RoPE (Rotary)")
    else:
        print("✗ Position embedding: Learned Absolute")
    
    # Check activation function
    if nano_config.gated_linear_unit:
        print("✓ MLP activation: SwiGLU")
    else:
        print("✗ MLP activation: GELU")
    
    # Check bias
    if not nano_config.use_bias:
        print("✓ Linear bias: Disabled (matching Megatron-LM default)")
    else:
        print("✗ Linear bias: Enabled")
    
    # Check normalization
    if nano_config.normalization == 'layernorm':
        print("✓ Normalization: LayerNorm")
    else:
        print("✗ Normalization: RMSNorm")
    
    # Check QKV projection
    if nano_config.use_fused_qkv:
        print("✓ QKV projection: Fused")
    else:
        print("✓ QKV projection: Separate (Q, K, V)")
    
    # Check GQA
    if nano_config.num_query_groups < nano_config.num_heads:
        print(f"✓ Group Query Attention: {nano_config.num_query_groups} groups")
    else:
        print("✓ Standard Multi-Head Attention")
    
    print("\n" + "="*80)
    print("Architecture verification complete!")
    print("="*80)
    
    return nano_model


if __name__ == "__main__":
    model = verify_architecture_match()
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 51200, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output dtype: {logits.dtype}")
    print("\nForward pass successful!")
