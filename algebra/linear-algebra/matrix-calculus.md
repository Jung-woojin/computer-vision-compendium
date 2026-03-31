# Matrix Calculus for CV/DL Researchers

## 1) 목적
논문 부록의 미분식을 "실제 코드 검증 도구"로 바꾸는 문서다.  
핵심은 다음 3가지:
- shape-safe한 미분 해석
- Jacobian/Hessian 직관
- gradient 디버깅 루틴

## 2) 필수 미분 규칙
- `d(a^T x) / dx = a`
- `d(x^T A x) / dx = (A + A^T)x` (A 대칭이면 `2Ax`)
- `d||Ax-b||_2^2 / dx = 2A^T(Ax-b)`
- `d tr(AXB) / dX = A^T B^T`
- softmax-cross-entropy 조합은 안정적 closed form gradient를 제공

## 3) 연구자 관점 / 엔지니어 관점

### 연구자 관점
- loss 구조를 matrix calculus로 쓰면 regularizer 의미가 명확해진다.
- Jacobian penalty는 representation smoothness 제어로 연결된다.
- Hessian 스펙트럼은 sharpness/generalization 관계를 분석하는 핵심 도구다.

### 엔지니어 관점
- autograd를 맹신하지 말고, 작은 텐서로 finite difference 검증을 한다.
- `requires_grad`, in-place 연산, detach 위치가 gradient 경로를 끊는지 확인한다.
- 커스텀 loss 작성 시 shape와 reduction(`mean/sum`)을 먼저 고정한다.

## 4) Jacobian 직관
- `J_f(x)`는 입력 주변에서 `f`를 선형화한 지도다.
- CV에서 의미:
  - 입력 perturbation 민감도 분석
  - adversarial robustness 진단
  - feature smoothness 정량화

실전 팁:
- 전체 Jacobian은 크므로 `||Jv||`, `||J^Tv||` 추정이 현실적이다.

## 5) Hessian 직관
- Hessian은 손실 지형의 곡률.
- 큰 고유값 방향: step이 크면 loss 폭증 가능.
- 작은/0 근처 방향: flat 혹은 saddle.

실전 근사:
- Hessian-vector product (HVP)
- trace 근사(Hutchinson)
- top eigenvalue (power iteration)

## 6) CV 논문에서 자주 등장하는 형태
- contrastive loss gradient 분해
- feature normalization의 미분 안정화 항
- second-order approximation으로 optimizer 해석
- curvature-aware training (SAM류)

## 7) 작은 실험 코드

```python
import torch

def finite_diff_grad(fn, x, eps=1e-4):
    g = torch.zeros_like(x)
    for i in range(x.numel()):
        x1 = x.clone().view(-1)
        x2 = x.clone().view(-1)
        x1[i] += eps
        x2[i] -= eps
        g.view(-1)[i] = (fn(x1.view_as(x)) - fn(x2.view_as(x))) / (2 * eps)
    return g

x = torch.randn(5, requires_grad=True)
A = torch.randn(5, 5)
b = torch.randn(5)

def fn(z):
    return torch.norm(A @ z - b) ** 2

loss = fn(x)
loss.backward()
g_auto = x.grad
g_fd = finite_diff_grad(fn, x.detach())
print("grad diff:", float(torch.norm(g_auto - g_fd)))
```

## 8) 디버깅 체크리스트
1. gradient가 None인 파라미터가 있는가?
2. gradient norm이 layer별로 폭발/소실하는가?
3. reduction 스케일링 때문에 학습률이 사실상 바뀌지 않았는가?
4. detach/in-place로 그래프가 끊기지 않았는가?
5. loss 항들의 단위(scale)가 지나치게 불균형하지 않은가?

## 9) 논문 읽기 연결
- 미분 부록을 읽을 때 "어떤 항이 불안정을 유발하는가"를 먼저 본다.
- Jacobian/Hessian 관련 주장은 반드시 계산 근사법과 함께 확인한다.
- 실험 섹션에서 gradient clipping/normalization이 등장하면 수학적 이유를 추적한다.

## 10) 후속 연결
- [fundamentals](C:/Users/ust21/linear-algebra-cv-dl-notes/linear-algebra/fundamentals.md)
- [eigens-and-svd](C:/Users/ust21/linear-algebra-cv-dl-notes/linear-algebra/eigens-and-svd.md)
- [optimization-geometry](C:/Users/ust21/linear-algebra-cv-dl-notes/linear-algebra/optimization-geometry.md)
