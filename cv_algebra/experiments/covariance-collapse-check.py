"""
Representation Collapse Detection via Covariance Spectrum

self-supervised learning 에서 representation collapse 를 조기 감지합니다.

Collapse signatures:
- Eigen spectrum이 한두 축으로 몰림
- Effective rank 급감
- Isotropy 위반 (균일성 상실)

Usage:
    python covariance-collapse-check.py --model resnet50 --eval-every 10
    python covariance-check.py --layer fc --save-spectrum
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def compute_covariance(features):
    """
    Batch features 의 covariance matrix 계산
    
    Args:
        features: (batch, features) shaped tensor
        
    Returns:
        covariance: (features, features) shaped matrix
    """
    # Center
    mean = features.mean(dim=0)
    X = features - mean
    
    # Covariance: (1/(N-1)) * X^T X
    n = features.shape[0]
    cov = (X.T @ X) / (n - 1)
    return cov


def compute_spectrum(cov):
    """
    Covariance matrix 의 고유값 스펙트럼 계산
    """
    eigenvalues = torch.linalg.eigvalsh(cov)
    # 음수 값 clamp (수치 불안정성)
    eigenvalues = torch.clamp(eigenvalues, min=0)
    return eigenvalues


def effective_rank(eigenvalues, epsilon=1e-6):
    """
    Effective rank 계산
    er = exp(-sum(p * log(p))) where p = eigenvalues / sum(eigenvalues)
    """
    total = eigenvalues.sum()
    p = eigenvalues / (total + epsilon)
    erank = torch.exp(-(p * torch.log(p + epsilon)).sum())
    return float(erank)


def spectral_bias(eigenvalues):
    """
    Spectral bias: 큰 고유값의 상대적 비중
    """
    total = eigenvalues.sum()
    top_ratio = eigenvalues.topk(min(3, len(eigenvalues)))[0].sum() / (total + epsilon)
    return float(top_ratio)


def isotropy_measure(eigenvalues):
    """
    Isotropy check: 모든 축이 균일한가?
    작은 값과 큰 값의 비율이 1 에 가까울수록 isotropic
    """
    total = eigenvalues.sum()
    p = eigenvalues / (total + 1e-12)
    entropy = -(p * torch.log(p + 1e-12)).sum()
    max_entropy = torch.log(torch.tensor(len(eigenvalues)))
    return float(entropy / (max_entropy + 1e-12))


def detect_collapse(eigenvalues, thresholds=None):
    """
    representation collapse 감지
    
    Return dict with:
    - collapse: bool
    - severity: str (none/warning/severe)
    - reason: list of failure modes
    """
    if thresholds is None:
        thresholds = {
            'effective_rank_min': 10,
            'top_ratio_max': 0.7,
            'isotropy_min': 0.5
        }
    
    erank = effective_rank(eigenvalues)
    top_ratio = spectral_bias(eigenvalues)
    isotropy = isotropy_measure(eigenvalues)
    
    reasons = []
    severity = "none"
    
    # Effective rank check
    if erank < thresholds['effective_rank_min']:
        reasons.append(f"effective_rank={erank:.1f} < {thresholds['effective_rank_min']}")
        severity = "warning"
    
    # Top eigenvalue ratio
    if top_ratio > thresholds['top_ratio_max']:
        reasons.append(f"top_k_ratio={top_ratio:.3f} > {thresholds['top_ratio_max']}")
        severity = "severe"
    
    # Isotropy
    if isotropy < thresholds['isotropy_min']:
        reasons.append(f"isotropy={isotropy:.3f} < {thresholds['isotropy_min']}")
        if severity != "severe":
            severity = "warning"
    
    return {
        'collapse': len(reasons) > 0,
        'severity': severity,
        'reasons': reasons,
        'metrics': {
            'effective_rank': erank,
            'top_ratio': top_ratio,
            'isotropy': isotropy
        }
    }


class FeatureCollector:
    """
    레이어 feature 를 수집하여 주기적으로 스펙트럼 분석
    """
    def __init__(self, model, layer_names, device="cpu"):
        self.model = model
        self.layer_names = layer_names
        self.device = device
        self.features = {name: [] for name in layer_names}
        self._hooks = {}
        self._register_hooks()
    
    def _register_hooks(self):
        for name in self.layer_names:
            hook = self._make_hook(name)
            self._hooks[name] = hook
    
    def _make_hook(self, layer_name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.features[layer_name].append(output.detach().cpu())
        return hook
    
    def start(self):
        for name in self.layer_names:
            layer = getattr(self.model, name, None)
            if layer:
                layer.register_forward_hook(self._hooks[name])
    
    def stop(self):
        for name in self.layer_names:
            if name in self._hooks:
                self._hooks[name].remove()
    
    def get_features(self):
        return {name: torch.cat(feats) for name, feats in self.features.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Detect representation collapse via covariance spectrum"
    )
    parser.add_argument(
        "--layer-names",
        type=str,
        default="fc,classifier,proj",
        help="Comma-separated layer names to monitor"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Collect features every N batches"
    )
    parser.add_argument(
        "--save-spectrum",
        action="store_true",
        help="Save eigenvalue spectra to disk"
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="JSON string of threshold overrides"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="spectrum_logs",
        help="Directory to save analysis results"
    )
    
    args = parser.parse_args()
    
    if args.thresholds:
        thresholds = json.loads(args.thresholds)
    else:
        thresholds = None
    
    print("=== Representation Collapse Detection ===")
    print(f"Monitoring layers: {args.layer_names.split(',')}")
    
    # 예제: random feature simulation
    print("\nSimulating feature collection...")
    
    layer_names = [name.strip() for name in args.layer_names.split(',')]
    batch_sizes = [64, 128, 256]
    feature_dims = [128, 256, 512, 1024]
    
    results = []
    
    for dim in feature_dims:
        for batch_size in batch_sizes:
            # Simulate different feature qualities
            feature_type = np.random.choice(['collapsed', 'isotropic', 'normal'])
            
            if feature_type == 'collapsed':
                # Collapse 시: 한두 개 축에 집중
                eigenvalues_sample = np.random.exponential(10, dim)
                eigenvalues_sample[0] *= 10
            elif feature_type == 'isotropic':
                # Isotropic: 모든 축 균등
                eigenvalues_sample = np.random.exponential(1, dim)
            else:
                # Normal: 중간 정도
                eigenvalues_sample = np.random.exponential(5, dim)
            
            eigenvalues = torch.tensor(eigenvalues_sample, dtype=torch.float32)
            
            detection = detect_collapse(eigenvalues, thresholds)
            detection['layer'] = "simulated"
            detection['dim'] = dim
            detection['batch_size'] = batch_size
            detection['feature_type'] = feature_type
            
            results.append(detection)
    
    # 결과 요약
    print("\n" + "="*60)
    print("COLLAPSE DETECTION SUMMARY")
    print("="*60)
    
    for r in results:
        status = "❌ COLLAPSE" if r['collapse'] else "✅ OK"
        print(f"\n{status} | dim={r['dim']:4d} | batch={r['batch_size']:4d} | type={r['feature_type']}")
        print(f"  Effective rank: {r['metrics']['effective_rank']:.1f}")
        print(f"  Top ratio: {r['metrics']['top_ratio']:.3f}")
        print(f"  Isotropy: {r['metrics']['isotropy']:.3f}")
        if r['reasons']:
            for reason in r['reasons']:
                print(f"  ⚠️  {reason}")
    
    # 디렉터리 생성 및 저장
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    import csv
    csv_path = output_path / "collapse_detection.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['layer', 'dim', 'batch_size', 'feature_type', 
                      'collapse', 'severity', 'effective_rank', 
                      'top_ratio', 'isotropy']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            writer.writerow({
                **r,
                **r['metrics']
            })
    
    print(f"\nResults saved to {csv_path}")
    
    if args.save_spectrum:
        import json
        spectrum_path = output_path / "spectra.json"
        spectra_data = []
        
        for r in results:
            layer = r['layer']
            dim = r['dim']
            feature_type = r['feature_type']
            
            # Simulate spectrum data
            eigenvalues_sample = np.random.exponential(5, dim)
            if feature_type == 'collapsed':
                eigenvalues_sample[0] *= 10
            
            spectra_data.append({
                'layer': layer,
                'dim': dim,
                'type': feature_type,
                'eigenvalues': eigenvalues_sample.tolist()[:100],  # truncate
                'detection': r
            })
        
        with open(spectrum_path, 'w') as f:
            json.dump(spectra_data, f, indent=2)
        
        print(f"Spectra saved to {spectrum_path}")


if __name__ == "__main__":
    main()
