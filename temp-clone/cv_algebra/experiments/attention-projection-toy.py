"""
Attention Projection Intuition - Toy Example

Query/Key projection 이 similarity computation 에 어떻게 영향을 미치는지
최소한의 코드직관적으로 보여줍니다.

Key concepts:
- Q = XW_Q, K = XW_K
- attention = softmax(QK^T / sqrt(d)) @ V
- projection 공간이 similarity 정의의 핵심

Usage:
    python attention-projection-toy.py --d_model 64 --n_heads 4
    python attention-projection-toy.py --compare-projections
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def setup_attention(d_model, n_heads, device="cpu"):
    """
    Simple multi-head attention setup
    """
    d_k = d_model // n_heads
    
    Q_proj = nn.Linear(d_model, d_model, bias=False)
    K_proj = nn.Linear(d_model, d_model, bias=False)
    V_proj = nn.Linear(d_model, d_model, bias=False)
    
    # Initialize
    nn.init.normal_(Q_proj.weight, mean=0, std=0.02)
    nn.init.normal_(K_proj.weight, mean=0, std=0.02)
    nn.init.normal_(V_proj.weight, mean=0, std=0.02)
    
    return Q_proj, K_proj, V_proj, d_k


def compute_attention_scores(X, Q_proj, K_proj, d_k, temperature=1.0, device="cpu"):
    """
    Compute Q, K, attention scores without softmax
    """
    B, N, D = X.shape
    
    Q = X @ Q_proj.weight.T  # (B, N, D)
    K = X @ K_proj.weight.T  # (B, N, D)
    
    # Attention scores: QK^T
    # Scale by sqrt(d_k)
    scale = 1.0 / np.sqrt(d_k)
    scores = (Q @ K.transpose(-2, -1)) * scale / temperature
    
    return Q, K, scores


def compare_projections(X, Q1, K1, Q2, K2, d_k, device="cpu"):
    """
    Compare two different projection pairs
    """
    B, N, D = X.shape
    
    Q1_, K1_, scores1 = compute_attention_scores(X, Q1, K1, d_k, device)
    Q2_, K2_, scores2 = compute_attention_scores(X, Q2, K2, d_k, device)
    
    return {
        'Q1': Q1_, 'K1': K1_, 'scores1': scores1,
        'Q2': Q2_, 'K2': K2_, 'scores2': scores2,
    }


def analyze_attention_similarity(scores):
    """
    Attention score distribution analysis
    """
    # Mean attention per token
    mean_per_token = scores.mean(dim=-1)
    
    # Entropy of attention (higher = more uniform)
    attn = torch.softmax(scores, dim=-1)
    entropy = -(attn * torch.log(attn + 1e-12)).sum(dim=-1)
    
    # Max attention (sparsity indicator)
    max_attn = attn.max(dim=-1).values
    
    return {
        'mean': float(mean_per_token.mean()),
        'std': float(mean_per_token.std()),
        'entropy': float(entropy.mean()),
        'max_attn': float(max_attn.mean()),
    }


def toy_projection_experiment():
    """
    Simple experiment: How different projections affect similarity
    """
    print("\n" + "="*60)
    print("ATTENTION PROJECTION INTUITION")
    print("="*60)
    
    # Setup
    d_model = 64
    n_heads = 4
    d_k = d_model // n_heads
    N = 10  # sequence length
    
    device = "cpu"
    X = torch.randn(1, N, d_model, device=device)
    
    Q1, K1, V1, _ = setup_attention(d_model, n_heads, device)
    Q2, K2, V2, _ = setup_attention(d_model, n_heads, device)
    
    print(f"\nSetup:")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_k per head: {d_k}")
    print(f"  sequence length: {N}")
    
    # Compute attention with first projection pair
    result1 = compute_attention_scores(X, Q1, K1, d_k, device)
    Q1_, K1_, scores1 = result1
    
    # Compute attention with second projection pair
    result2 = compute_attention_scores(X, Q2, K2, d_k, device)
    Q2_, K2_, scores2 = result2
    
    # Analyze similarity
    analysis1 = analyze_attention_similarity(scores1)
    analysis2 = analyze_attention_similarity(scores2)
    
    print("\n" + "-"*60)
    print("Analysis 1 (random projections)")
    print("-"*60)
    print(f"  Mean attention per token: {analysis1['mean']:.4f}")
    print(f"  Entropy (uniformity): {analysis1['entropy']:.4f}")
    print(f"  Max attention (sparsity): {analysis1['max_attn']:.4f}")
    
    print("\n" + "-"*60)
    print("Analysis 2 (different random projections)")
    print("-"*60)
    print(f"  Mean attention per token: {analysis2['mean']:.4f}")
    print(f"  Entropy (uniformity): {analysis2['entropy']:.4f}")
    print(f"  Max attention (sparsity): {analysis2['max_attn']:.4f}")
    
    # Key insight: projections define what similarity means
    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)
    print("""
The Q and K projections define the "similarity space" that attention uses.
    
1. Different projections → different similarity metrics
2. The projection space determines which tokens attend to each other
3. This is NOT non-linear - it's linear projection + similarity + weighted sum
    
Attention computation:
    Q = XW_Q, K = XW_K  (linear projections)
    similarity = QK^T / sqrt(d) (dot product similarity)
    attention = softmax(similarity) @ V (weighted sum)
    
The "magic" is in the learned projection matrices!
""")
    
    # Visual comparison
    print("\n" + "="*60)
    print("PRACTICAL CHECKLIST")
    print("="*60)
    print("""
1. ✓ Normalize position: before/after projection matters
2. ✓ Fixed similarity definition: don't mix cosine/dot product
3. ✓ Temperature scaling: controls attention sharpness
4. ✓ sqrt(d) scaling: prevents gradient vanishing
5. ✓ Consistency: use same similarity in code and analysis
""")
    
    # Show projection effect
    print("\n" + "="*60)
    print("PROJECTION EFFECT DEMONSTRATION")
    print("="*60)
    
    # Same X, different projections
    X_test = torch.randn(2, 5, d_model)
    Q_a, K_a, _, _ = setup_attention(d_model, n_heads, device)
    Q_b, K_b, _, _ = setup_attention(d_model, n_heads, device)
    
    scores_a = compute_attention_scores(X_test, Q_a, K_a, d_k, device)[2]
    scores_b = compute_attention_scores(X_test, Q_b, K_b, d_k, device)[2]
    
    print("\nTwo projection pairs on the same input:")
    print("  Score correlation: ", torch.corrcoef(scores_a.flatten(), scores_b.flatten())[0, 1].item())
    print("  → Different projections create different attention patterns!")
    
    return {
        'scores1': scores1,
        'scores2': scores2,
        'analysis1': analysis1,
        'analysis2': analysis2
    }


def advanced_comparison():
    """
    Advanced: Compare different projection strategies
    """
    print("\n" + "="*60)
    print("ADVANCED COMPARISON")
    print("="*60)
    
    d_model = 64
    N = 8
    device = "cpu"
    
    strategies = {
        'random': (torch.randn(d_model, d_model) @ torch.eye(d_model)),
        'identity': torch.eye(d_model),
        'diagonal': torch.diag(torch.randn(d_model)),
    }
    
    X = torch.randn(1, N, d_model, device=device)
    
    results = {}
    
    for name, W in strategies.items():
        Q_proj = nn.Linear(d_model, d_model, bias=False, device=device)
        K_proj = nn.Linear(d_model, d_model, bias=False, device=device)
        Q_proj.weight.data = W
        K_proj.weight.data = W
        
        Q, K, scores = compute_attention_scores(X, Q_proj, K_proj, d_model//4, device)
        
        analysis = analyze_attention_similarity(scores)
        results[name] = analysis
        
        print(f"\n{name}:")
        print(f"  Entropy: {analysis['entropy']:.4f}")
        print(f"  Max attention: {analysis['max_attn']:.4f}")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
Projection choice affects attention behavior:
- Random: diverse attention patterns
- Identity: preserves input similarities
- Diagonal: selective attention on features

Learning Q/K projections = learning the "similarity metric"
""")


def main():
    parser = argparse.ArgumentParser(
        description="Attention projection intuition - toy experiment"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=64,
        help="Model dimension"
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Run advanced comparison"
    )
    
    args = parser.parse_args()
    
    # Basic experiment
    results = toy_projection_experiment()
    
    # Advanced comparison
    if args.advanced:
        advanced_comparison()
    
    print("\n" + "="*60)
    print("CONNECTION TO LITERATURE")
    print("="*60)
    print("""
Related concepts in vision papers:

1. QK^T as Gram-like similarity
   - "attention map", "token affinity"
   - Cross-attention: modality alignment problem

2. Projection head
   - Linear projection after transformer layers
   - Improves representation quality

3. Temperature scaling
   - Controls attention sharpness
   - "temperature-scaled logits" in CLIP-like models

4. Normalization position
   - Before or after projection changes semantics
   - Fix normalization position consistently

Key papers:
- "Attention Is All You Need" (Vaswani et al.)
- "An Image is Worth 16x16 Words" (Dosovitskiy et al.)
- "CLIP" (Radford et al.)
- "Cross-attention in multimodal models"
""")


if __name__ == "__main__":
    main()
