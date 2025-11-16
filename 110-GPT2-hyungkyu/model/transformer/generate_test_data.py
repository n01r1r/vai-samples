"""
Generate reference test data for Transformer layers using NumPy
This creates known inputs and expected outputs for numerical verification
"""

import numpy as np
import json


def layer_norm(x, scale, shift, eps=1e-5):
    """
    Layer normalization

    Args:
        x: (batch, seq_len, d_model)
        scale: (d_model,)
        shift: (d_model,)
        eps: small constant for numerical stability

    Returns:
        output: (batch, seq_len, d_model)
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    norm_x = (x - mean) / np.sqrt(var + eps)
    return scale * norm_x + shift


def gelu(x):
    """
    GELU activation using tanh approximation

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    inner = sqrt_2_over_pi * (x + 0.044715 * np.power(x, 3))
    return 0.5 * x * (1.0 + np.tanh(inner))


def linear(x, weight):
    """Linear transformation: x @ weight.T"""
    return x @ weight.T


def feedforward(x, weight1, weight2):
    """
    Feed-forward network

    Args:
        x: (batch, seq_len, d_model)
        weight1: (4*d_model, d_model)
        weight2: (d_model, 4*d_model)

    Returns:
        output: (batch, seq_len, d_model)
        intermediates: dictionary with intermediate results
    """
    # Linear 1: d_model -> 4*d_model
    hidden = linear(x, weight1)

    # GELU activation
    gelu_out = gelu(hidden)

    # Linear 2: 4*d_model -> d_model
    output = linear(gelu_out, weight2)

    intermediates = {
        'hidden': hidden,
        'gelu_out': gelu_out
    }

    return output, intermediates


def multi_head_attention(x, W_q, W_k, W_v, W_out, num_heads):
    """
    Multi-head attention (simplified version from previous implementation)
    """
    batch, seq_len, d_model = x.shape
    head_dim = d_model // num_heads

    # Project to Q, K, V
    Q = linear(x, W_q)
    K = linear(x, W_k)
    V = linear(x, W_v)

    # Reshape for multi-head
    Q = Q.reshape(batch, seq_len, num_heads, head_dim)
    K = K.reshape(batch, seq_len, num_heads, head_dim)
    V = V.reshape(batch, seq_len, num_heads, head_dim)

    # Transpose to [B, H, S, HD]
    Q = Q.transpose(0, 2, 1, 3)
    K = K.transpose(0, 2, 1, 3)
    V = V.transpose(0, 2, 1, 3)

    # Attention scores
    attn_scores = Q @ K.transpose(0, 1, 3, 2)
    attn_scores = attn_scores / np.sqrt(head_dim)

    # Causal mask
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    attn_scores[:, :, mask] = -1e38

    # Softmax
    attn_scores_max = np.max(attn_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(attn_scores - attn_scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Weighted sum
    context = attn_weights @ V

    # Combine heads
    context = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

    # Output projection
    output = linear(context, W_out)

    return output


def transformer_block(x,
                     norm1_scale, norm1_shift,
                     W_q, W_k, W_v, W_out,
                     norm2_scale, norm2_shift,
                     ff_weight1, ff_weight2,
                     num_heads, eps=1e-5):
    """
    Transformer block with pre-norm architecture

    Args:
        x: (batch, seq_len, d_model)
        norm1_scale, norm1_shift: (d_model,)
        W_q, W_k, W_v: (d_model, d_model)
        W_out: (d_model, d_model)
        norm2_scale, norm2_shift: (d_model,)
        ff_weight1: (4*d_model, d_model)
        ff_weight2: (d_model, 4*d_model)
        num_heads: number of attention heads
        eps: LayerNorm epsilon

    Returns:
        output: (batch, seq_len, d_model)
    """
    # First residual block: x + Attention(LayerNorm(x))
    norm1_out = layer_norm(x, norm1_scale, norm1_shift, eps)
    attn_out = multi_head_attention(norm1_out, W_q, W_k, W_v, W_out, num_heads)
    residual1 = x + attn_out

    # Second residual block: residual1 + FeedForward(LayerNorm(residual1))
    norm2_out = layer_norm(residual1, norm2_scale, norm2_shift, eps)
    ff_out, _ = feedforward(norm2_out, ff_weight1, ff_weight2)
    output = residual1 + ff_out

    return output, {
        'norm1_out': norm1_out,
        'attn_out': attn_out,
        'residual1': residual1,
        'norm2_out': norm2_out,
        'ff_out': ff_out
    }


def generate_layer_norm_test():
    """Generate test case for LayerNorm"""
    print("\n" + "=" * 60)
    print("Generating LayerNorm Test Case")
    print("=" * 60)

    batch_size = 2
    seq_len = 3
    d_model = 8
    eps = 1e-5

    np.random.seed(42)

    # Input
    x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32) * 0.5

    # Parameters
    scale = np.ones(d_model, dtype=np.float32)
    shift = np.zeros(d_model, dtype=np.float32)

    # Forward pass
    output = layer_norm(x, scale, shift, eps)

    test_data = {
        'config': {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'd_model': d_model,
            'eps': eps
        },
        'input': x.tolist(),
        'scale': scale.tolist(),
        'shift': shift.tolist(),
        'output': output.tolist()
    }

    with open('layer_norm_test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"LayerNorm test data saved")
    print(f"  Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Output mean (should be ~0): {output.mean():.6f}")
    print(f"  Output std (should be ~1): {output.std():.6f}")


def generate_gelu_test():
    """Generate test case for GELU"""
    print("\n" + "=" * 60)
    print("Generating GELU Test Case")
    print("=" * 60)

    batch_size = 2
    seq_len = 3
    d_model = 8

    np.random.seed(42)

    # Input with diverse range
    x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32) * 2.0

    # Forward pass
    output = gelu(x)

    test_data = {
        'config': {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'd_model': d_model
        },
        'input': x.tolist(),
        'output': output.tolist()
    }

    with open('gelu_test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"GELU test data saved")
    print(f"  Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")


def generate_feedforward_test():
    """Generate test case for FeedForward"""
    print("\n" + "=" * 60)
    print("Generating FeedForward Test Case")
    print("=" * 60)

    batch_size = 2
    seq_len = 3
    d_model = 8
    hidden_dim = 4 * d_model  # 32

    np.random.seed(42)

    # Input
    x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32) * 0.5

    # Weights
    weight1 = np.random.randn(hidden_dim, d_model).astype(np.float32) * 0.1
    weight2 = np.random.randn(d_model, hidden_dim).astype(np.float32) * 0.1

    # Forward pass
    output, intermediates = feedforward(x, weight1, weight2)

    test_data = {
        'config': {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'd_model': d_model,
            'hidden_dim': hidden_dim
        },
        'input': x.tolist(),
        'weight1': weight1.tolist(),
        'weight2': weight2.tolist(),
        'intermediates': {
            'hidden': intermediates['hidden'].tolist(),
            'gelu_out': intermediates['gelu_out'].tolist()
        },
        'output': output.tolist()
    }

    with open('feedforward_test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"FeedForward test data saved")
    print(f"  Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  Hidden range: [{intermediates['hidden'].min():.4f}, {intermediates['hidden'].max():.4f}]")
    print(f"  GELU output range: [{intermediates['gelu_out'].min():.4f}, {intermediates['gelu_out'].max():.4f}]")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")


def generate_transformer_block_test():
    """Generate test case for TransformerBlock"""
    print("\n" + "=" * 60)
    print("Generating TransformerBlock Test Case")
    print("=" * 60)

    batch_size = 2
    seq_len = 3
    d_model = 8
    num_heads = 2
    hidden_dim = 4 * d_model
    eps = 1e-5

    np.random.seed(42)

    # Input
    x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32) * 0.5

    # LayerNorm1 parameters
    norm1_scale = np.ones(d_model, dtype=np.float32)
    norm1_shift = np.zeros(d_model, dtype=np.float32)

    # Attention weights
    W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
    W_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
    W_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
    W_out = np.random.randn(d_model, d_model).astype(np.float32) * 0.1

    # LayerNorm2 parameters
    norm2_scale = np.ones(d_model, dtype=np.float32)
    norm2_shift = np.zeros(d_model, dtype=np.float32)

    # FeedForward weights
    ff_weight1 = np.random.randn(hidden_dim, d_model).astype(np.float32) * 0.1
    ff_weight2 = np.random.randn(d_model, hidden_dim).astype(np.float32) * 0.1

    # Forward pass
    output, intermediates = transformer_block(
        x,
        norm1_scale, norm1_shift,
        W_q, W_k, W_v, W_out,
        norm2_scale, norm2_shift,
        ff_weight1, ff_weight2,
        num_heads, eps
    )

    test_data = {
        'config': {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'd_model': d_model,
            'num_heads': num_heads,
            'hidden_dim': hidden_dim,
            'eps': eps
        },
        'input': x.tolist(),
        'weights': {
            'norm1_scale': norm1_scale.tolist(),
            'norm1_shift': norm1_shift.tolist(),
            'W_query': W_q.tolist(),
            'W_key': W_k.tolist(),
            'W_value': W_v.tolist(),
            'W_out': W_out.tolist(),
            'norm2_scale': norm2_scale.tolist(),
            'norm2_shift': norm2_shift.tolist(),
            'ff_weight1': ff_weight1.tolist(),
            'ff_weight2': ff_weight2.tolist()
        },
        'intermediates': {
            'norm1_out': intermediates['norm1_out'].tolist(),
            'attn_out': intermediates['attn_out'].tolist(),
            'residual1': intermediates['residual1'].tolist(),
            'norm2_out': intermediates['norm2_out'].tolist(),
            'ff_out': intermediates['ff_out'].tolist()
        },
        'output': output.tolist()
    }

    with open('transformer_block_test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"TransformerBlock test data saved")
    print(f"  Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")


if __name__ == "__main__":
    generate_layer_norm_test()
    generate_gelu_test()
    generate_feedforward_test()
    generate_transformer_block_test()
    print("\n" + "=" * 60)
    print("All test data generation complete!")
    print("=" * 60)
