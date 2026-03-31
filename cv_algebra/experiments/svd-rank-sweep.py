"""
SVD-based Rank Compression Sweep

행렬 SVD 를 통해 rank-k 근사의 정확도-효율 trade-off 를 측정합니다.

Usage:
    python svd-rank-sweep.py --layers linear1,linear2,mlp
    python svd-rank-sweep.py --epochs 10 --rank-sweep 4,8,16,32,64
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def compress_linear(W, k):
    """
    행렬 W 를 rank-k 로 근사합니다.
    
    Args:
        W: 원본 행렬 (out_features, in_features)
        k: 목표 rank
        
    Returns:
        A, B: W ≈ A @ B, A=(out_features, k), B=(k, in_features)
    """
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    A = U[:, :k] @ torch.diag(S[:k])
    B = Vh[:k, :]
    return A, B


def compute_effective_rank(S, epsilon=1e-6):
    """
    effective rank 계산
    er = exp(-sum(p * log(p))) where p = S / sum(S)
    """
    p = S / (S.sum() + epsilon)
    erank = torch.exp(-(p * torch.log(p + epsilon)).sum())
    return float(erank)


def compress_and_evaluate(model_name, W, ranks, device="cpu"):
    """
    rank sweep 실행 후 결과 반환
    """
    results = []
    
    # 전체 SVD
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    original_norm = torch.norm(W)
    effective_rank = compute_effective_rank(S)
    
    print(f"\n{model_name}")
    print(f"  Shape: {W.shape}")
    print(f"  Effective rank: {effective_rank:.1f}")
    print(f"  Parameters: {W.numel():,}")
    
    for k in ranks:
        A, B = compress_linear(W, k)
        Wk = A @ B
        rel_err = torch.norm(W - Wk) / (original_norm + 1e-12)
        
        # 파라미터 비율
        params_ratio = (A.numel() + B.numel()) / W.numel()
        
        # 연산량 비율 (행렬곱 FLOPs 비율)
        flops_ratio = (k * W.shape[0] + k * W.shape[1]) / (W.shape[0] * W.shape[1])
        
        results.append({
            'k': k,
            'rel_err': float(rel_err),
            'params_ratio': float(params_ratio),
            'flops_ratio': float(flops_ratio)
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SVD rank compression sweep for vision models"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="linear1,linear2,mlp",
        help="Comma-separated layer names to test"
    )
    parser.add_argument(
        "--rank-sweep",
        type=str,
        default="16,32,64,128,256,512",
        help="Comma-separated rank values to test"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda)"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Path to save results CSV"
    )
    
    args = parser.parse_args()
    
    ranks = [int(x) for x in args.rank_sweep.split(',')]
    devices = ["cuda" if torch.cuda.is_available() else "cpu"]
    device = args.device if args.device in devices else devices[0]
    
    print(f"Using device: {device}")
    
    # 실험용 데이터 생성 (실제 모델 weight 로 대체 가능)
    test_cases = {
        "attention_q": torch.randn(1024, 512, device=device),
        "attention_v": torch.randn(1024, 512, device=device),
        "attention_k": torch.randn(1024, 512, device=device),
        "mlp_down": torch.randn(4096, 2048, device=device),
        "mlp_up": torch.randn(2048, 8192, device=device),
        "conv1x1": torch.randn(256, 128, 1, 1, device=device),
    }
    
    all_results = []
    
    for layer_name, W in test_cases.items():
        # Conv weight to linear
        if len(W.shape) == 4:
            W = W.view(W.shape[0], -1)
        
        results = compress_and_evaluate(layer_name, W, ranks, device)
        all_results.extend(results)
        
        # Top 3 rank 추천
        sorted_results = sorted(results, key=lambda x: x['rel_err'])
        print(f"  Top 3 ranks: {[r['k'] for r in sorted_results[:3]]}")
    
    # 결과 요약 출력
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Rank':<8}{'Avg Rel Err':<12}{'Params Ratio':<15}{'FLOPs Ratio':<15}")
    print("-"*60)
    
    for k in sorted(set(r['k'] for r in all_results)):
        ks_results = [r for r in all_results if r['k'] == k]
        avg_err = np.mean([r['rel_err'] for r in ks_results])
        avg_params = np.mean([r['params_ratio'] for r in ks_results])
        avg_flops = np.mean([r['flops_ratio'] for r in ks_results])
        print(f"{k:<8}{avg_err:<12.6f}{avg_params:<15.3f}{avg_flops:<15.3f}")
    
    # CSV 저장
    if args.save_results:
        import csv
        with open(args.save_results, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['layer', 'k', 'rel_err', 'params_ratio', 'flops_ratio'])
            writer.writeheader()
            for layer_name, W in test_cases.items():
                if len(W.shape) == 4:
                    layer_name = f"{layer_name}_viewed"
                for r in all_results:
                    writer.writerow({
                        'layer': layer_name,
                        **r
                    })
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
