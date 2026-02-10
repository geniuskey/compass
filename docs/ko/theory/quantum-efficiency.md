# 양자 효율

양자 효율(Quantum Efficiency, QE)은 COMPASS에서 계산하는 핵심 성능 지표(Figure of Merit)입니다. 픽셀이 입사 광자를 전기 신호로 얼마나 효과적으로 변환하는지를 측정합니다.

## 정의

파장 $\lambda$에서의 외부 양자 효율은 다음과 같습니다:

$$\text{QE}(\lambda) = \frac{\text{수집된 전자-정공 쌍의 수}}{\text{입사 광자의 수}}$$

광학 전력으로 등가 표현하면:

$$\text{QE}(\lambda) = \frac{P_\text{absorbed in PD}(\lambda)}{P_\text{incident}(\lambda)}$$

여기서 $P_\text{absorbed in PD}$는 포토다이오드 체적 내에서 흡수된 전력이고, $P_\text{incident}$는 총 입사 전력입니다. QE는 무차원량으로, 0에서 1 사이(0% ~ 100%)의 값을 가집니다.

## QE에 영향을 미치는 요소

전체 QE는 여러 손실 요소의 곱입니다:

$$\text{QE} = (1 - R) \times T_\text{optics} \times \eta_\text{abs} \times \text{FF}_\text{PD}$$

| 요소 | 기호 | 설명 |
|--------|--------|-------------|
| 표면 반사 | $R$ | 픽셀 스택에 의해 반사되는 빛 (BARL이 이를 감소시킴) |
| 광학 투과율 | $T_\text{optics}$ | 컬러 필터, 평탄화층 등을 통과하는 비율 |
| 실리콘 흡수 | $\eta_\text{abs}$ | 실리콘 두께 내에서 흡수되는 빛의 비율 |
| 포토다이오드 충전율 | $\text{FF}_\text{PD}$ | PD 수집 체적 내에서 흡수된 광자의 비율 |

COMPASS에서는 전파 시뮬레이션이 이 모든 효과를 동시에 포착합니다. QE는 위의 분해된 형태가 아니라 전체 전자기 해석 결과로부터 계산됩니다.

## COMPASS에서 QE 계산

COMPASS는 두 가지 방법으로 QE를 계산합니다:

### 방법 1: 흡수 적분

위치 $\mathbf{r}$, 파장 $\lambda$에서의 흡수 전력 밀도(Absorbed Power Density)는 다음과 같습니다:

$$p_\text{abs}(\mathbf{r}) = \frac{1}{2} \omega \varepsilon_0 \text{Im}(\varepsilon_r) |\mathbf{E}(\mathbf{r})|^2$$

특정 픽셀(포토다이오드 영역 $V_\text{PD}$)에 대한 QE는:

$$\text{QE} = \frac{\int_{V_\text{PD}} p_\text{abs} \, dV}{P_\text{incident}}$$

### 방법 2: 포인팅 플럭스 차이

대안으로, 특정 영역에서 흡수된 전력은 영역에 출입하는 순 포인팅 플럭스(Net Poynting Flux)로부터 구할 수 있습니다:

$$P_\text{absorbed} = \oint_S \langle \mathbf{S} \rangle \cdot \hat{n} \, dA = S_{z,\text{top}} - S_{z,\text{bottom}}$$

여기서 $S_{z,\text{top}}$과 $S_{z,\text{bottom}}$은 포토다이오드 영역의 상부와 하부에서의 포인팅 벡터 z 성분입니다.

두 방법 모두 `QECalculator` 클래스에 구현되어 있습니다.

## 픽셀별 QE와 색상 채널

2x2 베이어 단위 셀에는 각각 고유한 포토다이오드를 가진 네 개의 픽셀이 있습니다:

```
  +--------+--------+
  | R_0_0  | G_0_1  |
  +--------+--------+
  | G_1_0  | B_1_1  |
  +--------+--------+
```

COMPASS는 각 포토다이오드에 대해 QE를 독립적으로 계산합니다. 명명 규칙은 `{색상}_{행}_{열}`입니다.

`spectral_response` 함수는 동일한 색상의 픽셀에 대해 QE를 평균하여 채널별 QE 곡선을 생성합니다:

```python
from compass.analysis.qe_calculator import QECalculator

# result.qe_per_pixel = {"R_0_0": array, "G_0_1": array, "G_1_0": array, "B_1_1": array}
channel_qe = QECalculator.spectral_response(result.qe_per_pixel, result.wavelengths)
# channel_qe = {"R": (wavelengths, qe_R), "G": (wavelengths, qe_G_avg), "B": (wavelengths, qe_B)}
```

## 크로스토크

광학적 크로스토크(Optical Crosstalk)는 한 픽셀에 입사해야 할 빛이 인접 픽셀에 흡수되는 현상입니다. COMPASS는 이를 **크로스토크 행렬(Crosstalk Matrix)**로 정량화합니다:

$$\text{CT}_{ij}(\lambda) = \frac{\text{QE}_j(\lambda, \text{illuminating pixel } i)}{\sum_k \text{QE}_k(\lambda, \text{illuminating pixel } i)}$$

대각 요소는 정확하게 검출된 신호를, 비대각 요소는 크로스토크를 나타냅니다. 크로스토크가 낮을수록 색 분리가 우수합니다.

`QECalculator.compute_crosstalk` 메서드는 픽셀별 QE 데이터로부터 이 행렬을 계산합니다.

<CrosstalkHeatmap />

## 에너지 보존

기본적인 물리적 제약 조건은 에너지 보존(Energy Conservation)입니다:

$$R(\lambda) + T(\lambda) + A(\lambda) = 1$$

여기서 $R$은 전체 반사율, $T$는 전체 투과율(하부를 통한), $A$는 전체 흡수율(모든 재료에서)입니다. 이 균형이 1~2% 이상 위반되면 시뮬레이션에 수치적 문제가 있을 수 있습니다.

모든 픽셀의 총 QE는 실리콘에서의 전체 흡수율에 의해 상한이 결정됩니다:

$$\sum_\text{pixels} \text{QE}_i \leq A_\text{Si}$$

이 부등식이 엄격한 이유는 일부 빛이 컬러 필터, 금속 격자, 기타 비포토다이오드 영역에서 흡수되기 때문입니다.

## 일반적인 QE 스펙트럼

잘 설계된 1 um 피치 BSI 픽셀의 경우:

| 채널 | 최대 QE | 최대 QE 파장 |
|---------|---------|-----------------|
| 파란색 | 50-70% | 450-470 nm |
| 녹색 | 60-80% | 530-560 nm |
| 빨간색 | 50-70% | 600-630 nm |

QE 스펙트럼의 일반적인 특징:
- 청색 가장자리에서 실리콘 흡수 증가에 따른 급격한 상승
- 각 컬러 필터 통과 대역에서의 피크
- 적색 가장자리에서 실리콘 흡수 감소(흡수 깊이가 픽셀 두께를 초과)에 따른 점진적 감소
- BARL 스택의 박막 간섭에 의한 스펙트럼 리플

::: info
단일 채널에서 80%를 초과하는 QE는 드문데, 이는 컬러 필터 흡수, 반사 손실, 포토다이오드 충전율 등이 모두 전체 효율을 감소시키기 때문입니다.
:::
