# CMOS 이미지 센서 (CIS) 기술 트렌드 종합 조사

> 조사일: 2026-02-11 | COMPASS 프로젝트 참고용

---

## 1. 픽셀 피치 스케일링 역사

### 1.1 주요 마일스톤

CMOS 이미지 센서의 픽셀 피치는 지난 25년간 약 20배 축소되었다. 아래 표는 주요 시기별 대표 픽셀 피치와 기술적 전환점을 정리한 것이다.

| 시기 | 대표 픽셀 피치 | 해상도 범위 | 핵심 기술 전환 | 대표 제품/기업 |
|------|---------------|------------|---------------|---------------|
| 2000-2004 | 5.6-12 um | 1-5 MP | FSI 3T/4T APS, 180nm 공정 | Canon D30 (2000), Sony ICX282 |
| 2005-2007 | 2.2-5.6 um | 5-12 MP | 컬럼 ADC, 공유 구조 (shared pixel) | Sony Exmor (2007), 컬럼 A/D 상용화 |
| 2008-2010 | 1.4-2.2 um | 8-16 MP | **BSI 상용화**, gapless 마이크로렌즈 | Sony BSI (2009), OmniVision OmniBSI |
| 2011-2014 | 1.1-1.4 um | 13-20 MP | Stacked BSI, PDAF (Phase Detection AF) | Sony Exmor RS (2012), 적층 구조 시작 |
| 2015-2017 | 0.9-1.12 um | 16-48 MP | DTI (Deep Trench Isolation), Dual PD | Samsung ISOCELL (2013~), Sony Dual PD |
| 2018-2020 | 0.7-0.9 um | 48-108 MP | Quad Bayer/Nonacell, Full DTI | Samsung HP1 108MP, Sony IMX586 48MP |
| 2021-2023 | 0.56-0.7 um | 108-200 MP | CDTI, 3-layer stacking, meta-optics | Samsung HP3 (0.56um), OV OVB0A (0.56um) |
| 2024-2026+ | 0.5-0.56 um | 200 MP+ | QQBC, on-sensor AI, 2-layer transistor | Sony 200MP QQBC (2025), 차세대 연구 |

### 1.2 물리적 한계

현재 0.56 um 픽셀 피치는 녹색광 파장(~0.55 um)과 거의 동일한 수준이다. 이 영역에서 발생하는 근본적 제약 사항:

- **회절 한계 (Diffraction limit):** 픽셀 크기가 파장에 근접하면 Airy disk이 인접 픽셀을 침범하여 광학적 분해능이 저하됨
- **광자 수 감소 (Photon starvation):** 픽셀 면적 축소에 따라 수집 가능한 광자 수가 줄어 SNR (Signal-to-Noise Ratio) 저하
- **Full-well capacity 감소:** 포토다이오드 부피 축소로 최대 저장 전하량 감소 → 다이나믹 레인지 제한
- **읽기 잡음 (Read noise):** 0.15 e- rms 수준의 Skipper-in-CMOS 기술이 등장했으나 회로 복잡도 증가
- **렌즈 요구사항:** f-number가 작은 (빠른) 렌즈가 필요하여 모듈 설계 복잡도 증가

이러한 한계로 인해 업계는 단순한 픽셀 미세화보다 **적층 아키텍처 + 계산 사진학 (computational photography)**으로 전략을 전환하고 있다.

---

## 2. BSI (Backside Illumination) vs FSI (Frontside Illumination)

### 2.1 구조적 차이

```
FSI (Frontside Illumination)         BSI (Backside Illumination)
 ┌─────────────────┐                 ┌─────────────────┐
 │   마이크로렌즈    │                 │   마이크로렌즈    │
 │   컬러 필터      │                 │   컬러 필터      │
 │   ─────────────  │                 │   포토다이오드    │  ← 빛이 직접 도달
 │   배선층 (M1-M5) │  ← 빛 차단      │   ─────────────  │
 │   포토다이오드    │                 │   배선층 (M1-M5) │
 │   Si 기판        │                 │   Si 기판 (박화)  │
 └─────────────────┘                 └─────────────────┘
       ↓ 빛                                ↓ 빛
```

FSI 구조에서는 입사광이 다층 금속 배선(Metal interconnect)을 통과한 후에야 포토다이오드에 도달한다. 배선층에서의 반사, 산란, 차폐로 인해 유효 개구율 (fill factor)이 크게 저하된다. 특히 픽셀 피치가 2 um 이하로 축소되면 배선과 포토다이오드의 면적 경쟁이 심화된다.

### 2.2 BSI가 승리한 이유

| 항목 | FSI | BSI | 개선율 |
|------|-----|-----|--------|
| 양자 효율 (QE) @550nm | ~40-60% | ~80-90%+ | 1.5-2x |
| 개구율 (Fill factor) | 30-60% | ~100% | 1.5-3x |
| CRA 허용 범위 | 넓음 | 제한적 (개선중) | - |
| 제조 비용 | 낮음 | 높음 (웨이퍼 박화) | - |
| 기생 광감도 (Parasitic light) | 높음 | 낮음 | 크게 개선 |
| 크로스토크 @1.0um pitch | 15-25% | 5-10% | 2-3x |

### 2.3 BSI 제조 공정 핵심

1. **웨이퍼 박화 (Wafer thinning):** Si 기판을 ~3 um 이하로 연마/식각 → 기계적 강도 확보가 과제
2. **캐리어 웨이퍼 접합 (Carrier wafer bonding):** 박화 전 지지 기판 접합으로 핸들링 가능
3. **TSV (Through-Silicon Via):** 뒷면에서 앞면 회로로의 전기적 연결
4. **후면 처리 (Backside processing):** 반사 방지 코팅 (ARC), 컬러 필터, 마이크로렌즈를 후면에 형성

### 2.4 현재 상태

2025년 기준, 모바일/소비자용 CIS는 사실상 **100% BSI**로 전환 완료. 글로벌 BSI CIS 시장은 2025년 약 150억 달러 규모이며, 2033년까지 연평균 7.4% 이상 성장이 전망된다. Sony, Samsung, OmniVision 3사가 시장을 주도하고 있다.

---

## 3. 적층형 (Stacked) 센서

### 3.1 적층 기술 진화

적층형 CIS는 센서의 광전 변환부와 신호 처리 회로를 별도 웨이퍼에 제작한 후 접합하는 기술이다.

| 세대 | 구조 | 시기 | 핵심 특징 | 대표 제품 |
|------|------|------|----------|----------|
| 1세대 | 2-layer (Pixel + Logic) | 2012~ | 픽셀 웨이퍼와 로직 웨이퍼 분리 | Sony Exmor RS (IMX135) |
| 2세대 | 2-layer + DRAM | 2017~ | 고속 읽기용 DRAM 적층 | Sony IMX400 (Xperia XZs) |
| 3세대 | 3-layer (Pixel + Logic + Memory) | 2021~ | 완전 3층 적층, Global Shutter | Sony IMX900, Exmor T |
| 4세대 | 2-layer transistor pixel | 2021~ | 포토다이오드/트랜지스터 층 분리 | Sony 2-layer transistor (2021 발표) |
| 5세대 | 3-layer + AI processor | 2025~ | On-sensor AI 처리 | Sony 200MP + AI (2025 발표) |

### 3.2 접합 기술 (Bonding Technology)

적층형 센서의 핵심은 웨이퍼 간 접합 기술이다:

- **Cu-Cu hybrid bonding:** 구리 패드 직접 접합. 피치 ~3-5 um 수준까지 미세화 진행. 현재 업계 주류
- **Oxide bonding:** SiO2 면 접합 후 TSV로 전기 연결. 초기 세대에 사용
- **Micro-bump:** 솔더 범프 기반 접합. 접합 피치가 커서 최신 제품에는 부적합
- **Pixel-level connection:** Cu-Cu 하이브리드 본딩의 미세화로 픽셀 단위 수직 연결 가능 → 각 픽셀이 독립적으로 하부 로직에 접근

### 3.3 Sony의 3-layer 전략

Sony Semiconductor Solutions의 CEO Shinji Sahsida는 2025년 투자자 프레젠테이션에서 **3층 적층 센서의 장기 로드맵**을 공개했다:

- **포토다이오드 층과 트랜지스터 층을 분리** 최적화하여 포화 신호량(saturation signal level)을 기존 대비 약 2배 향상
- **다이나믹 레인지 확대:** HDR 성능 획기적 개선
- **멀티모달 센싱 + on-chip AI:** 이미지 처리를 넘어 지능형 센싱으로 전환

### 3.4 Samsung의 대응

Samsung은 2026년경 Apple iPhone용 **3-layer 적층 CIS** 양산을 목표로 개발 중이다. 1/2.6인치 48MP 초광각 센서부터 시작하여 Sony의 장기 독점 공급 체제에 도전할 전망이다.

---

## 4. DTI (Deep Trench Isolation) 진화

### 4.1 DTI 기술의 필요성

픽셀 피치가 축소될수록 인접 픽셀 간 **광학적 크로스토크 (optical crosstalk)**와 **전기적 크로스토크 (electrical crosstalk)**가 급격히 증가한다. DTI는 실리콘 기판에 깊은 트렌치를 식각하고 절연 재료로 충전하여 픽셀 간 격리를 구현하는 핵심 기술이다.

### 4.2 DTI 세대별 진화

| 세대 | 기술 명칭 | 트렌치 깊이 | 충전 재료 | 크로스토크 개선 | 적용 시기 |
|------|----------|------------|----------|---------------|----------|
| 0세대 | STI (Shallow Trench) | <0.5 um | SiO2 | 기준선 | ~2008 |
| 1세대 | Partial BDTI (Backside DTI) | 1-2 um | SiO2 | 30-50% 감소 | 2012~ |
| 2세대 | Full DTI (F-DTI) | Full depth (~3 um) | SiO2 / Poly-Si | 60-80% 감소 | 2015~ |
| 3세대 | Metal-filled DTI | Full depth | W (텅스텐) + liner | 80-90% 감소, 광차폐 | 2018~ |
| 4세대 | CDTI (Capacitive DTI) | Full depth | MOS capacitor 구조 | 90%+ 감소, FWC 향상 | 2020~ |
| 5세대 | Air-gap DTI | Full depth | Air gap + ARC | 최대 격리, 반사 활용 | 연구 단계 |

### 4.3 CDTI (Capacitive DTI) 상세

CDTI는 트렌치 벽면에 MOS 커패시터 구조를 형성하여 다음과 같은 장점을 제공한다:

- **극저 암전류 (Dark current):** 60C에서 1.4 um 픽셀 기준 약 1 aA (아토암페어) 수준
- **높은 FWC (Full-well Capacity):** ~12,000 e- (1.4 um 픽셀)로 기존 DTI 대비 향상
- **양자 효율 개선:** 측벽 임플란트 없이도 QE 향상 달성
- **핀닝 전압 제어:** 트렌치 측벽의 전위를 능동적으로 제어하여 공핍 영역 최적화

### 4.4 DTI + BSI 시너지

Full DTI가 BSI와 결합될 때 최대 효과를 발휘한다:

```
         빛 입사 ↓
    ┌──────┬──────┬──────┐
    │ ML   │ ML   │ ML   │  마이크로렌즈
    │ CF   │ CF   │ CF   │  컬러 필터
    ├──┤   ├──┤   ├──┤   │
    │  │PD │  │PD │  │PD │  포토다이오드
    │  │   │  │   │  │   │
    │D │   │D │   │D │   │
    │T │   │T │   │T │   │  DTI (Full depth)
    │I │   │I │   │I │   │
    │  │   │  │   │  │   │
    ├──┤   ├──┤   ├──┤   │
    │  배선층  │  배선층  │  배선층  │
    └──────┴──────┴──────┘
```

이 구조에서 DTI는 광학적 도파관 (optical waveguide) 역할을 하며, 빛을 포토다이오드 내부에 가두어 흡수 효율을 극대화한다. 특히 NIR (Near-Infrared) 영역에서 효과가 크다.

### 4.5 자동차용 F-DTI

자동차 이미지 센서에서는 **2.1 um Full-Depth DTI** 공정이 적용되어, 85C 접합 온도에서 단일 노출 120 dB 다이나믹 레인지를 달성한 사례가 보고되었다. 스토리지 커패시터를 픽셀 내에 통합하여 HDR 성능과 기능 안전 (functional safety) 요구를 동시에 충족한다.

---

## 5. 마이크로렌즈 기술

### 5.1 마이크로렌즈의 역할

마이크로렌즈는 각 픽셀 상부에 위치하여 입사광을 포토다이오드 유효 영역으로 집광하는 핵심 광학 소자이다. 픽셀 피치 축소에 따라 마이크로렌즈 설계의 중요성이 급격히 증가한다.

### 5.2 기술 진화

| 세대 | 기술 | 특징 | 시기 |
|------|------|------|------|
| 1세대 | 구형 (Spherical) 마이크로렌즈 | 단순 반구형, 픽셀 간 간격 존재 | ~2005 |
| 2세대 | Gapless 마이크로렌즈 | 인접 렌즈 간 간격 제거, 집광 효율 향상 | 2008~ |
| 3세대 | 비구면 (Aspherical) 마이크로렌즈 | Superellipse 프로파일, CRA 최적화 | 2012~ |
| 4세대 | 다층 (Multi-layer) 마이크로렌즈 | 내부 렌즈 + 외부 렌즈 조합 | 2016~ |
| 5세대 | 메타옵틱스 (Meta-optics) 마이크로렌즈 | 나노구조 기반 평면 렌즈, 파장 선택적 집광 | 연구 단계 |

### 5.3 Superellipse 프로파일

COMPASS에서 사용하는 마이크로렌즈 모델:

```
z(x, y) = h * (1 - r^2)^(1/2a)

여기서:
  h = 렌즈 높이 (sag height)
  r = 정규화된 반경 (0 ~ 1)
  a = superellipse 파라미터 (a=1: 반구, a<1: 평탄화, a>1: 가파름)
```

이 프로파일은 CRA (Chief Ray Angle)에 따른 집광 효율 최적화에 핵심적이다. 높은 CRA 설계에서는 비대칭 프로파일이 필요하며, 이는 시뮬레이션에서 정확히 재현해야 할 중요한 요소이다.

### 5.4 CRA (Chief Ray Angle) 최적화

CRA는 렌즈 시스템의 주광선이 센서 표면에 입사하는 각도이다:

- **Low CRA (<15 deg):** 빛이 수직에 가깝게 입사 → 마이크로렌즈 설계 용이, QE 균일
- **High CRA (>25 deg):** 센서 가장자리에서 빛이 비스듬히 입사 → 마이크로렌즈 시프트 (shift) 및 비대칭 프로파일 필요
- **CRA 매칭:** 모바일 카메라 모듈에서는 렌즈 CRA와 센서 CRA를 정밀하게 매칭해야 주변부 감광 (vignetting) 최소화

최근 2024년 연구에서는 **Adjoint sensitivity analysis** 기반 마이크로렌즈 형상 최적화가 제안되어, 2회의 전자기 시뮬레이션만으로 figure of merit의 기울기를 계산할 수 있다.

### 5.5 메타옵틱스 (Meta-optics) 마이크로렌즈

서브마이크론 픽셀 시대에 기존 굴절형 마이크로렌즈의 성능 한계를 극복하기 위해 메타옵틱스 기술이 연구되고 있다:

- **나노필라 (Nanopillar) 어레이:** 서브파장 크기의 기둥 구조로 위상 제어
- **파장 선택적 집광:** 컬러 필터 기능과 집광 기능을 단일 층에서 통합 가능
- **편광 감도:** 특정 편광 상태에 대한 선택적 응답 구현 가능

---

## 6. 컬러 필터 기술

### 6.1 전통적 Bayer 패턴과 그 진화

| 패턴 | 배열 | 유효 해상도 | 빈닝 (Binning) | 대표 적용 |
|------|------|-----------|----------------|----------|
| Bayer (RGGB) | 2x2 | 1x | 2x2 → 1/4 | 대부분의 CIS |
| Quad Bayer (QQBC) | 4x4 동색 | 1x or 1/4x | 4x4 → 1/16 | Sony IMX586+ |
| Nona-Bayer (Nonacell) | 3x3 동색 | 1x or 1/9x | 3x3 → 1/9 | Samsung HP1 108MP |
| Hexadeca Bayer | 4x4 동색 | 1x or 1/16x | 이중 빈닝 가능 | Samsung HP3 200MP |
| QQBC (Quad-Quad) | 16px 클러스터 | 1x or 1/16x | 16→1 초고감도 | Sony 200MP (2025) |

### 6.2 Quad Bayer / Nona-Bayer의 장점

동일 색상의 인접 픽셀 클러스터를 형성함으로써:

1. **저조도 모드:** N개 픽셀 신호를 합산 (binning)하여 감도 N배 향상
2. **고해상도 모드:** 각 픽셀을 독립적으로 읽어 전체 해상도 활용
3. **위상 검출 AF:** 동색 픽셀 쌍으로 위상차 검출 가능 (Super QPD)
4. **HDR:** 동색 픽셀 내 다른 노출 시간 설정으로 단일 프레임 HDR

### 6.3 Sony QQBC (2025)

Sony가 2025년 발표한 200MP 센서는 **Quad-Quad Bayer Coding (QQBC)** 배열을 채택했다:

- 0.7 um 픽셀 피치, 1/1.12인치 포맷
- 16개(4x4) 동색 픽셀 클러스터 → 야간/실내에서 16픽셀 합산 고감도
- **On-sensor AI** 내장: 해상도 복원, 노이즈 저감을 센서 칩 내에서 처리
- 단안 (monocular) 카메라에서 고배율 줌 시 고화질 유지

### 6.4 유기 컬러 필터 (Organic Color Filter)

기존 안료/염료 기반 컬러 필터의 한계를 극복하기 위한 연구:

- **유기 광전 변환막 (OPD, Organic Photodetector):** 특정 파장 선택 흡수
- **적층 유기 센서:** R, G, B 각 층을 수직으로 적층 → Bayer 패턴 불필요, 전체 픽셀에서 전색 정보 획득
- **Foveon 방식의 한계 극복:** 유기물의 흡수 스펙트럼 엔지니어링으로 색 분리 개선
- **Panasonic/Fujifilm:** 유기 CMOS 센서 연구 지속 (상용화는 제한적)

### 6.5 무기 양자점 컬러 필터 (QD Color Filter)

양자점의 크기 의존적 흡수/발광 특성을 활용:

- 양자점 크기(2-10 nm)로 흡수 파장 정밀 제어
- 기존 안료 필터 대비 좁은 흡수 대역 → 색 순도 향상 가능
- SWIR (Short-Wave Infrared) 확장 용이: PbS 양자점으로 900-1700 nm 감지

---

## 7. 광전변환 효율 (QE) 최적화

### 7.1 QE 결정 요인

양자 효율은 입사 광자가 전자-정공 쌍으로 변환되는 비율이며, 다음 요소들의 곱으로 결정된다:

```
QE_total = T_lens * T_filter * (1 - R_surface) * eta_absorption * eta_collection

여기서:
  T_lens      = 마이크로렌즈 투과율
  T_filter    = 컬러 필터 투과율
  R_surface   = 표면 반사율
  eta_absorption = 실리콘 내 광흡수 효율 (두께, 파장 의존)
  eta_collection = 광생성 캐리어 수집 효율
```

### 7.2 반사 방지 코팅 (Anti-Reflection Coating, ARC)

| 기술 | 반사율 | 파장 범위 | 특징 |
|------|--------|----------|------|
| 단층 ARC (SiN) | ~5% | 좁음 | 단순, 저비용 |
| 다층 ARC (SiO2/SiN/HfO2) | ~1-2% | 400-700 nm | 표준 기술 |
| 나노구조 ARC (moth-eye) | <0.5% | 300-1000 nm | 광대역, 공정 복잡 |
| 3D 나노콘 (Nanocone) | <1% | 가시광 | 유연 기판 적용 가능, EQE 7% 향상 보고 |

최신 연구(2023)에서 표면 나노엔지니어링을 적용한 상용 BSI CIS에서 **300-700 nm 범위에서 90% 이상의 QE**를 달성한 바 있다. 이 기술은 동시에 암전류를 3배 저감하는 부가 효과도 보였다.

### 7.3 광 포획 구조 (Light-Trapping Structures)

서브마이크론 픽셀에서 얇은 실리콘 두께로 인한 흡수 부족(특히 NIR)을 보완하기 위한 구조:

- **역피라미드 어레이 (Inverted Pyramid Array, IPA):** BSI CIS 뒷면에 2D 주기 구조 형성. 1.2 um 픽셀, 400 nm 피치 IPA에서 850 nm 파장 감도 **80% 향상** 보고
- **단일 홀 (Single hole) 구조:** 포토다이오드 상의 최적 크기 홀이 NIR 흡수를 **60% 향상** (3 um Si 기준)
- **회절 격자 (Diffraction grating):** 후면 회절 구조로 광 경로 연장 → 실효 흡수 두께 증가
- **DTI 광도파관:** Full DTI 벽면에서의 전반사로 빛을 픽셀 내부에 가두는 효과

### 7.4 에너지 보존 검증

COMPASS에서의 핵심 물리적 제약 조건:

```
R + T + A = 1  (허용 오차 < 1%)

여기서:
  R = 반사율 (Reflectance)
  T = 투과율 (Transmittance)
  A = 흡수율 (Absorptance) ≥ QE (일부 흡수는 열로 소산)
```

시뮬레이션에서 이 에너지 보존 관계가 성립하지 않으면 수치 오류를 의심해야 한다. 특히 RCWA에서 Fourier order 부족 시 에너지 비보존이 발생할 수 있다.

---

## 8. 차세대 기술

### 8.1 양자점 이미지 센서 (Quantum Dot Image Sensor)

2024년은 **"양자점의 해 (Year of the Quantum Dot)"**로 불릴 만큼 양자점 센서 기술이 급진전했다:

- **PbS 양자점 SWIR 센서:** 실리콘 ROIC (Readout IC)와 모놀리식 집적. 자율주행, 식품 검사, 의료 영상에 적용 확대
- **Global shutter 동작:** PbS QD SWIR 센서에서 글로벌 셔터 동작 시연
- **핵심 성능:** 검출도 (Detectivity) >4.2x10^17 Jones, 응답도 >8.3x10^3 A/W, 검출 범위 365-1310 nm
- **환경 친화적 합성:** 폐 납산 배터리에서 추출한 PbS QD로 센서 제작 연구 (2024)

### 8.2 유기 광검출기 (Organic Photodetector, OPD)

유기 반도체 기반 광전 변환 소자:

- **장점:** 흡수 스펙트럼 튜닝 용이, 대면적 코팅 가능, 유연 기판 적용
- **수직 적층 색분리:** R/G/B 유기층을 수직으로 적층하여 Bayer 패턴 없이 전색 정보 획득
- **과제:** CMOS 공정 호환성, 장기 안정성, 암전류 제어
- **Perovskite 광검출기:** 유기-무기 하이브리드 소재로 높은 흡수 계수와 긴 캐리어 수명

### 8.3 이벤트 기반 비전 센서 (Event-Based Vision Sensor, EVS)

Sony와 Prophesee의 협업으로 상용화가 진전된 EVS/DVS 기술:

- **동작 원리:** 각 픽셀이 비동기적으로 휘도 변화를 감지하여 변화가 발생한 픽셀의 좌표와 시간만 출력
- **Sony IMX636:** 1280x720 HD, 업계 최소 4.86 um 픽셀, 다이나믹 레인지 86 dB+ (5-100,000 lux)
- **최대 이벤트 레이트:** 1.06 Giga-events/sec
- **장점:** 극저 지연 (<1 us), 저전력, 저대역폭 (변화 없는 장면에서는 데이터 없음)
- **적용:** 산업 검사, 자율주행, 로봇 비전, 제스처 인식

IDS는 2025년 Sony-Prophesee IMX636 기반 산업용 이벤트 카메라 시리즈 (uEye XCP-E)를 출시했다.

### 8.4 계산 이미징 (Computational Imaging)

하드웨어와 소프트웨어의 융합으로 기존 센서의 물리적 한계를 극복:

- **On-sensor AI:** Sony의 200MP QQBC 센서에 AI 프로세서 내장 → 센서 단에서 노이즈 저감, 해상도 복원
- **Multi-frame 합성:** HDR, 저조도 촬영에서 다중 프레임 합성으로 SNR 향상
- **Depth 센싱:** ToF (Time-of-Flight), structured light과 CIS의 통합
- **Neural ISP:** 전통적 ISP 파이프라인을 신경망으로 대체 → 단대단 최적화

---

## 9. 시뮬레이션 관점: COMPASS의 역할과 과제

### 9.1 기술 트렌드가 시뮬레이션에 미치는 영향

위에서 다룬 기술 트렌드 각각이 COMPASS와 같은 광학 시뮬레이션 도구에 새로운 요구사항을 부과한다:

| 기술 트렌드 | 시뮬레이션 요구사항 | COMPASS 대응 |
|-----------|-------------------|-------------|
| 서브마이크론 픽셀 | 파동 광학 (RCWA/FDTD) 필수, 기하 광학 부적합 | RCWA 다중 솔버 (torcwa, grcwa, meent) |
| BSI 구조 | 다층 박막 간섭 정확 모델링 | PixelStack 레이어 기반 구조 |
| 적층 센서 | 복잡한 3D 구조, 대규모 계산 | GPU 가속 (PyTorch/JAX) |
| Full DTI | 금속/유전체 트렌치의 정확한 유전율 모델링 | MaterialDB (내장 + CSV + 분산 모델) |
| 마이크로렌즈 | Superellipse 프로파일, CRA 의존성 | GeometryBuilder, cone illumination |
| 컬러 필터 | 재료 분산 (n, k vs wavelength) | Cauchy/Sellmeier 피팅 |
| QE 최적화 | 에너지 보존 검증 (R+T+A=1) | energy_balance.py |
| 메타옵틱스 | 서브파장 구조 → 높은 Fourier order 필요 | RCWA stability (S-matrix) |

### 9.2 COMPASS가 해결하는 핵심 문제

1. **솔버 간 교차 검증 (Cross-solver validation):** 동일 구조에 대해 RCWA (torcwa, grcwa, meent)와 FDTD (flaport) 결과를 비교하여 시뮬레이션 신뢰도 확보
2. **단일 YAML 설정:** 복잡한 픽셀 구조를 선언적으로 정의 → 재현성과 파라미터 스윕 용이
3. **솔버 독립적 추상화:** PixelStack → 솔버별 변환을 자동화하여 새로운 솔버 추가가 용이
4. **물리적 일관성 자동 검증:** QE 범위 (0-1), 에너지 보존 (R+T+A=1) 자동 체크

### 9.3 향후 시뮬레이션 과제

서브 0.5 um 픽셀 시대를 대비하여 COMPASS가 대응해야 할 과제:

- **계산 비용 관리:** 3D FDTD의 경우 서브마이크론 픽셀에서도 계산 도메인이 크지 않으나, 파장 스윕과 파라미터 최적화 시 수천 회 반복이 필요
- **역설계 (Inverse design) 통합:** 자동 미분 (AD) 기반 마이크로렌즈/메타옵틱스 최적화 → meent, fmmax 등 AD 지원 솔버의 중요성 증가
- **다중 물리 연계:** 광학 시뮬레이션 + 전하 수송 (carrier transport) + 회로 시뮬레이션의 통합
- **양자점/유기 재료 모델링:** 새로운 광전 변환 재료의 복소 유전율 데이터베이스 확장 필요
- **이벤트 센서 시뮬레이션:** 시간 영역 응답 모델링 → FDTD와의 자연스러운 연계 가능성

### 9.4 시뮬레이션 정확도 기준

COMPASS에서의 시뮬레이션 결과 신뢰성을 위한 핵심 검증 기준:

| 검증 항목 | 기준값 | 비고 |
|----------|--------|------|
| 에너지 보존 (R+T+A) | = 1 (오차 < 1%) | 모든 파장에서 확인 |
| Si 굴절률 @550nm | n ~ 4.08, k ~ 0.028 | Green 2008 데이터 기준 |
| QE 범위 | 0 <= QE <= 1 | 물리적 상한/하한 |
| 솔버 간 QE 편차 | < 5% (동일 조건) | RCWA vs FDTD 교차 검증 |
| RCWA 수렴 | Fourier order 증가 시 수렴 확인 | S-matrix만 사용 (T-matrix 불가) |
| 크로스토크 | 인접 픽셀 신호 누설 비율 | DTI 유무에 따른 차이 검증 |

---

## 참고 자료 및 출처

### 산업 보고서
- Yole Group, "Status of the CMOS Image Sensor Industry 2025"
- Mordor Intelligence, "Image Sensors Market Size, Trends, Share Analysis 2030"
- IDTechEx, "Quantum Dots Revolutionizing Image Sensors" (2024)

### 기업 기술 문서
- Sony Semiconductor Solutions, QQBC 기술 및 200MP 센서 발표 (2025)
- Samsung Semiconductor, ISOCELL HP3 0.56um 픽셀 기술 문서 (2022)
- OmniVision, OmniBSI 기술 백서
- Prophesee/Sony, IMX636 Event-Based Vision Sensor 사양서

### 학술 논문
- "CMOS Image Sensor for Broad Spectral Range with >90% Quantum Efficiency" (Small, 2023)
- "Deep Trench Isolation and Inverted Pyramid Array Structures for CMOS Image Sensor" (Sensors, 2020)
- "Automotive 2.1um Full-Depth DTI CMOS Image Sensor with 120dB Dynamic Range" (Sensors, 2023)
- "Adjoint-Assisted Shape Optimization of Microlenses for CMOS Image Sensors" (PMC, 2024)
- "IR Sensitivity Enhancement of CMOS Image Sensor with Diffractive Light Trapping Pixels" (Scientific Reports, 2017)

### 업계 발표 및 뉴스
- IEEE Spectrum, "Samsung and OmniVision Claim Smallest Camera Pixels" (2022)
- SK hynix Newsroom, "Evolution of Pixel Technology in CMOS Image Sensor"
- DPReview, "Tech Timeline: Milestones in Sensor Development"
- Image Sensors World (imagesensors.org), IISW 워크숍 논문들

---

> 본 문서는 COMPASS 프로젝트의 기술적 맥락을 제공하기 위한 조사 자료이며, 시뮬레이션 파라미터 설정 및 검증 기준의 근거로 활용된다. 기술 데이터는 2026년 2월 기준이며, 급변하는 CIS 산업 특성상 정기적 업데이트가 필요하다.
