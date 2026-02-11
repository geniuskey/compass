# 핵심 학술 논문 정리 (Key Academic Papers)

COMPASS 프로젝트의 이론적, 실무적 기반을 형성하는 핵심 논문들의 주석 목록이다.
각 논문에 대해 전체 인용 정보, 한국어 요약, 관련성 태그를 포함한다.

---

## 1. RCWA 기초 이론

COMPASS의 핵심 솔버인 RCWA(Rigorous Coupled-Wave Analysis)의 이론적 기초를 제공하는 논문들이다.

### 1.1 Moharam & Gaylord (1981) -- RCWA 최초 정립

- **인용**: M. G. Moharam and T. K. Gaylord, "Rigorous coupled-wave analysis of planar-grating diffraction," *J. Opt. Soc. Am.*, vol. 71, no. 7, pp. 811--818, 1981. DOI: [10.1364/JOSA.71.000811](https://doi.org/10.1364/JOSA.71.000811)
- **요약**: 주기 구조에서 전자기파 회절을 수치적으로 엄밀하게 풀기 위한 결합파 해석법(RCWA)을 최초로 정립한 논문이다. 유전율의 푸리에 급수 전개와 고유값 문제를 결합하여 임의 회절격자에 대한 정확한 투과/반사 효율을 계산하는 방법을 제시했다. COMPASS에서 사용하는 모든 RCWA 솔버(torcwa, grcwa, meent)의 이론적 출발점이다.
- **태그**: [RCWA] [기초이론]

### 1.2 Moharam & Gaylord (1986) -- 금속 표면 릴리프 격자

- **인용**: M. G. Moharam and T. K. Gaylord, "Rigorous coupled-wave analysis of metallic surface-relief gratings," *J. Opt. Soc. Am. A*, vol. 3, no. 11, pp. 1780--1787, 1986. DOI: [10.1364/JOSAA.3.001780](https://doi.org/10.1364/JOSAA.3.001780)
- **요약**: RCWA를 금속 표면 릴리프 격자에 확장 적용한 논문으로, 복소 유전율을 가진 재료에 대해 TE/TM 편광 및 임의 입사각 처리를 가능하게 했다. COMPASS에서 텅스텐 그리드나 금속 차광막 등 흡수성 금속 구조를 시뮬레이션할 때 필수적인 이론 기반을 제공한다.
- **태그**: [RCWA] [금속구조]

### 1.3 Moharam et al. (1995a) -- 안정적 구현과 S-matrix

- **인용**: M. G. Moharam, E. B. Grann, D. A. Pommet, and T. K. Gaylord, "Formulation for stable and efficient implementation of the rigorous coupled-wave analysis of binary gratings," *J. Opt. Soc. Am. A*, vol. 12, no. 5, pp. 1068--1076, 1995. DOI: [10.1364/JOSAA.12.001068](https://doi.org/10.1364/JOSAA.12.001068)
- **요약**: RCWA의 수치 안정성 문제를 체계적으로 분석하고 S-matrix 알고리즘을 도입한 논문이다. T-matrix 방식의 지수 오버플로우 문제를 해결하는 Redheffer star product 기반 접근법을 제안했다. COMPASS의 `StableSMatrixAlgorithm` 모듈이 이 논문의 방법론을 직접 구현한다.
- **태그**: [RCWA] [수치안정성]

### 1.4 Moharam et al. (1995b) -- 향상된 투과 행렬 접근법

- **인용**: M. G. Moharam, D. A. Pommet, E. B. Grann, and T. K. Gaylord, "Stable implementation of the rigorous coupled-wave analysis for surface-relief gratings: enhanced transmittance matrix approach," *J. Opt. Soc. Am. A*, vol. 12, no. 5, pp. 1077--1086, 1995. DOI: [10.1364/JOSAA.12.001077](https://doi.org/10.1364/JOSAA.12.001077)
- **요약**: 앞선 논문의 후속편으로, 향상된 투과 행렬(ETM) 접근법을 상세히 제시한다. S-matrix와 ETM 두 가지 안정적 구현 방법을 비교 분석하여, 다층 구조 시뮬레이션에서의 수치 안정성을 보장하는 실용적 가이드를 제공한다.
- **태그**: [RCWA] [수치안정성]

### 1.5 Li (1996a) -- 푸리에 급수의 불연속 함수 처리

- **인용**: L. Li, "Use of Fourier series in the analysis of discontinuous periodic structures," *J. Opt. Soc. Am. A*, vol. 13, no. 9, pp. 1870--1876, 1996. DOI: [10.1364/JOSAA.13.001870](https://doi.org/10.1364/JOSAA.13.001870)
- **요약**: 불연속 주기 구조의 푸리에 급수 해석에서 발생하는 수렴 문제를 수학적으로 규명한 논문이다. 불연속 함수의 곱에 대한 올바른 푸리에 인수분해 규칙(Li rules)을 정립했으며, 이는 금속/유전체 경계에서의 TM 편광 수렴성을 획기적으로 개선한다. COMPASS의 `LiFactorization.convolution_matrix_inverse_rule` 메서드의 이론적 근거이다.
- **태그**: [RCWA] [수렴성] [Li규칙]

### 1.6 Li (1996b) -- 산란 행렬 재귀 알고리즘 비교

- **인용**: L. Li, "Formulation and comparison of two recursive matrix algorithms for modeling layered diffraction gratings," *J. Opt. Soc. Am. A*, vol. 13, no. 5, pp. 1024--1035, 1996. DOI: [10.1364/JOSAA.13.001024](https://doi.org/10.1364/JOSAA.13.001024)
- **요약**: 다층 회절 격자 모델링을 위한 두 가지 재귀 행렬 알고리즘(S-matrix와 R-matrix)을 정립하고 비교한 논문이다. S-matrix 방식이 수치적으로 무조건 안정적(unconditionally stable)임을 증명했으며, COMPASS가 S-matrix만을 사용하는 근거를 제공한다.
- **태그**: [RCWA] [수치안정성] [S-matrix]

### 1.7 Li (1997) -- 교차 표면 릴리프 격자의 FMM

- **인용**: L. Li, "New formulation of the Fourier modal method for crossed surface-relief gratings," *J. Opt. Soc. Am. A*, vol. 14, no. 10, pp. 2758--2767, 1997. DOI: [10.1364/JOSAA.14.002758](https://doi.org/10.1364/JOSAA.14.002758)
- **요약**: 2차원 교차 격자에 대한 푸리에 모달법(FMM)의 새로운 정식화를 제안한 논문이다. 올바른 푸리에 인수분해 규칙을 2D 구조에 적용하여 수렴 속도를 크게 향상시켰다. COMPASS에서 2x2 베이어 패턴 등 2D 주기 구조를 다룰 때 핵심적인 알고리즘 기반이다.
- **태그**: [RCWA] [2D격자] [수렴성]

### 1.8 Lalanne (1997) -- 결합파 방법의 개선된 정식화

- **인용**: P. Lalanne, "Improved formulation of the coupled-wave method for two-dimensional gratings," *J. Opt. Soc. Am. A*, vol. 14, no. 7, pp. 1592--1598, 1997. DOI: [10.1364/JOSAA.14.001592](https://doi.org/10.1364/JOSAA.14.001592)
- **요약**: Li의 인수분해 규칙과 독립적으로, 2D 격자에 대한 결합파 방법의 수렴성을 개선하는 새로운 정식화를 제안했다. 유전체, 금속, 체적 및 표면 릴리프 격자 등 다양한 회절 문제에서 기존 방법 대비 빠른 수렴을 수치적으로 입증했다.
- **태그**: [RCWA] [수렴성] [2D격자]

### 1.9 Popov & Neviere (2001) -- 고속 수렴 정식화

- **인용**: E. Popov and M. Neviere, "Maxwell equations in Fourier space: fast-converging formulation for diffraction by arbitrary shaped, periodic, anisotropic media," *J. Opt. Soc. Am. A*, vol. 18, no. 11, pp. 2886--2894, 2001. DOI: [10.1364/JOSAA.18.002886](https://doi.org/10.1364/JOSAA.18.002886)
- **요약**: 임의 형태의 주기 구조에 대해 푸리에 공간에서 맥스웰 방정식의 고속 수렴 정식화를 제안한 논문이다. 빠른 푸리에 인수분해(Fast Fourier Factorization, FFF) 방법을 통해 비정형 경계에서의 수렴 속도를 개선했으며, 비등방성 매질까지 확장 가능하다.
- **태그**: [RCWA] [수렴성] [FFF]

---

## 2. FDTD 기초 이론

COMPASS의 대안 솔버인 FDTD(Finite-Difference Time-Domain)의 이론적 기초를 형성하는 논문들이다.

### 2.1 Yee (1966) -- FDTD 방법 최초 제안

- **인용**: K. S. Yee, "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media," *IEEE Trans. Antennas Propag.*, vol. 14, no. 3, pp. 302--307, 1966. DOI: [10.1109/TAP.1966.1138693](https://doi.org/10.1109/TAP.1966.1138693)
- **요약**: 맥스웰 방정식의 시간 영역 유한차분법을 최초로 제안한 기념비적 논문이다. 공간적으로 엇갈린(staggered) 격자 위에서 전기장과 자기장을 교대로 계산하는 Yee 격자를 도입했다. COMPASS에서 flaport FDTD 솔버의 근본 알고리즘이다.
- **태그**: [FDTD] [기초이론]

### 2.2 Taflove & Hagness (2005) -- FDTD 교과서

- **인용**: A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed. Norwood, MA: Artech House, 2005. ISBN: 978-1580538329
- **요약**: FDTD 방법론에 대한 가장 포괄적인 참고 교과서이다. 기본 알고리즘부터 흡수 경계 조건, 분산 매질 모델링, 근거리-원거리장 변환까지 FDTD의 모든 측면을 다룬다. COMPASS의 FDTD 솔버 구현에서 수치 분산, 안정성 조건(CFL), 격자 해상도 기준 등의 참고 자료로 사용된다.
- **태그**: [FDTD] [교과서]

### 2.3 Berenger (1994) -- PML 흡수 경계 조건

- **인용**: J.-P. Berenger, "A perfectly matched layer for the absorption of electromagnetic waves," *J. Comput. Phys.*, vol. 114, no. 2, pp. 185--200, 1994. DOI: [10.1006/jcph.1994.1159](https://doi.org/10.1006/jcph.1994.1159)
- **요약**: 전자기파 시뮬레이션에서 반사 없이 파동을 흡수하는 완벽 정합층(PML)을 최초로 제안한 논문이다. FDTD 시뮬레이션의 개방 경계 조건 문제를 혁신적으로 해결했으며, 이후 모든 EM 시뮬레이터의 표준 경계 조건이 되었다. COMPASS의 FDTD 솔버에서 계산 영역 종단 처리에 필수적이다.
- **태그**: [FDTD] [PML] [경계조건]

### 2.4 Gedney (1996) -- 단축 PML (UPML)

- **인용**: S. D. Gedney, "An anisotropic perfectly matched layer-absorbing medium for the truncation of FDTD lattices," *IEEE Trans. Antennas Propag.*, vol. 44, no. 12, pp. 1630--1639, 1996. DOI: [10.1109/8.546249](https://doi.org/10.1109/8.546249)
- **요약**: Berenger의 분할장 PML을 단축 비등방성 매질(Uniaxial PML)로 재정식화한 논문이다. 물리적으로 더 직관적이며 구현이 간단한 UPML 방식은 현대 FDTD 코드에서 널리 채택되었다. 분산 매질 및 비등방성 매질에 대한 PML 확장에도 유리하다.
- **태그**: [FDTD] [PML] [UPML]

---

## 3. 실리콘 광학 상수

이미지 센서 시뮬레이션의 핵심 재료인 실리콘의 광학 상수에 관한 논문들이다.

### 3.1 Green (2008) -- 실리콘 광학 파라미터 표준 데이터

- **인용**: M. A. Green, "Self-consistent optical parameters of intrinsic silicon at 300 K including temperature coefficients," *Sol. Energy Mater. Sol. Cells*, vol. 92, no. 11, pp. 1305--1310, 2008. DOI: [10.1016/j.solmat.2008.06.009](https://doi.org/10.1016/j.solmat.2008.06.009)
- **요약**: 300K에서의 진성 실리콘 광학 상수(흡수 계수, 굴절률, 소멸 계수)를 0.25--1.45um 파장 범위에서 자기일관성 있게 정리한 표준 데이터이다. Kramers-Kronig 분석을 기반으로 업데이트된 반사율 데이터에서 도출했으며, 온도 계수도 포함한다. COMPASS MaterialDB의 실리콘 데이터 기준이 되는 논문이다 (550nm에서 n~4.08, k~0.028).
- **태그**: [재료] [실리콘] [광학상수]

<MaterialBrowser />

### 3.2 Aspnes & Studna (1983) -- 분광 타원법 데이터

- **인용**: D. E. Aspnes and A. A. Studna, "Dielectric functions and optical parameters of Si, Ge, GaP, GaAs, GaSb, InP, InAs, and InSb from 1.5 to 6.0 eV," *Phys. Rev. B*, vol. 27, no. 2, pp. 985--1009, 1983. DOI: [10.1103/PhysRevB.27.985](https://doi.org/10.1103/PhysRevB.27.985)
- **요약**: 분광 타원법(spectroscopic ellipsometry)을 사용하여 Si를 포함한 8개 반도체의 유전 함수와 광학 파라미터를 1.5--6.0eV 범위에서 측정한 논문이다. 3,100회 이상 인용된 고영향 논문으로, 반도체 광학 상수의 표준 참고 자료이다.
- **태그**: [재료] [실리콘] [분광법]

### 3.3 Palik (1998) -- 광학 상수 핸드북

- **인용**: E. D. Palik, ed., *Handbook of Optical Constants of Solids*, Academic Press, 1998. ISBN: 978-0125444156
- **요약**: SiO2, Si3N4, HfO2, TiO2, 텅스텐 등 COMPASS에서 사용하는 대부분의 재료에 대한 광학 상수를 포괄적으로 수록한 참고 서적이다. 넓은 파장 범위(자외선~적외선)에 걸친 n, k 값을 제공하며, COMPASS MaterialDB의 내장 재료 데이터 검증에 사용된다.
- **태그**: [재료] [광학상수] [참고서]

---

## 4. 이미지 센서 광학

CMOS 이미지 센서(CIS) 픽셀의 광학 시뮬레이션에 관한 핵심 논문들이다.

### 4.1 Catrysse & Wandell (2002) -- CIS 광학 효율 분석

- **인용**: P. B. Catrysse and B. A. Wandell, "Optical efficiency of image sensor pixels," *J. Opt. Soc. Am. A*, vol. 19, no. 8, pp. 1610--1620, 2002. DOI: [10.1364/JOSAA.19.001610](https://doi.org/10.1364/JOSAA.19.001610)
- **요약**: CMOS 이미지 센서 픽셀의 광학 효율을 기하광학적 위상 공간 접근법으로 분석한 선구적 논문이다. 실험 측정과 3% 이내의 정확도로 일치하는 이론적 예측을 제시했으며, 이후 FDTD 기반 CIS 시뮬레이션 연구의 기초를 놓았다. COMPASS의 QE 계산 및 크로스토크 분석 방법론의 출발점이다.
- **태그**: [CIS] [QE] [광학효율]

### 4.2 Agranov, Berezin & Tsai (2003) -- 크로스토크와 마이크로렌즈

- **인용**: G. Agranov, V. Berezin, and R. H. Tsai, "Crosstalk and microlens study in a color CMOS image sensor," *IEEE Trans. Electron Devices*, vol. 50, no. 1, pp. 4--11, 2003. DOI: [10.1109/TED.2002.806473](https://doi.org/10.1109/TED.2002.806473)
- **요약**: 컬러 CMOS 이미지 센서에서 광학 크로스토크와 마이크로렌즈의 역할을 실험적으로 연구한 논문이다. 픽셀 크기 축소에 따른 크로스토크 증가 문제를 정량적으로 분석하고, 마이크로렌즈 최적화를 통한 개선 가능성을 제시했다. COMPASS에서 마이크로렌즈 시프트와 크로스토크 분석의 근거가 된다.
- **태그**: [CIS] [크로스토크] [마이크로렌즈]

### 4.3 El Gamal & Eltoukhy (2005) -- CIS 리뷰

- **인용**: A. El Gamal and H. Eltoukhy, "CMOS image sensors," *IEEE Circuits and Devices Magazine*, vol. 21, no. 3, pp. 6--20, 2005. DOI: [10.1109/MCD.2005.1438751](https://doi.org/10.1109/MCD.2005.1438751)
- **요약**: CMOS 이미지 센서의 동작 원리, 아키텍처, 성능 지표를 포괄적으로 리뷰한 논문이다. 광 변환 효율, 노이즈 원인, 픽셀 회로 설계 등 CIS의 전반적 이해를 위한 필수 참고 자료이다.
- **태그**: [CIS] [리뷰]

### 4.4 Yokogawa et al. (2017) -- BSI 센서의 광 트래핑

- **인용**: S. Yokogawa, I. Oshiyama, H. Ikeda, Y. Ebiko, T. Hirano, S. Saito, T. Oinoue, Y. Hagimoto, and H. Iwamoto, "IR sensitivity enhancement of CMOS Image Sensor with diffractive light trapping pixels," *Sci. Rep.*, vol. 7, 3832, 2017. DOI: [10.1038/s41598-017-04200-y](https://doi.org/10.1038/s41598-017-04200-y)
- **요약**: BSI(후면 조사) CMOS 이미지 센서에서 2차원 역 피라미드 배열 구조(IPA)와 DTI(Deep Trench Isolation)를 활용한 회절형 광 트래핑 기술을 보고한 논문이다. 850nm에서 80%의 감도 향상을 달성했으며, FDTD 시뮬레이션과 실험 데이터의 비교 분석을 제시한다. COMPASS의 DTI 및 BSI 픽셀 시뮬레이션의 중요한 검증 참고 자료이다.
- **태그**: [CIS] [BSI] [광트래핑] [DTI]

### 4.5 Hwang & Kim (2023) -- 비축 픽셀의 광학 스택 정렬

- **인용**: J.-H. Hwang and Y. Kim, "A Numerical Method of Aligning the Optical Stacks for All Pixels," *Sensors*, vol. 23, no. 2, 702, 2023. DOI: [10.3390/s23020702](https://doi.org/10.3390/s23020702)
- **요약**: 비축 픽셀에 대해 레이어별 광학 스택 시프트(마이크로렌즈, 컬러 필터, 패시베이션)를 계산하는 스넬 법칙 기반 폐쇄형 방법을 제안한 논문이다. 쿼드 CF와 인픽셀 DTI가 적용된 서브마이크론 BSI 픽셀(0.5-1.0 um)에서 CRA 0-35° 범위로 검증하였으며, FDTD 대비 오차 <4.2%를 달성했다. COMPASS는 `PixelStack._compute_snell_shift()`에서 스넬 법칙 누적 시프트를 구현한다.
- **태그**: [CIS] [마이크로렌즈] [CRA] [광학정렬]

---

## 5. 수치 안정성

RCWA 시뮬레이션의 수치 안정성 확보에 관한 논문들이다.

### 5.1 Schuster et al. (2007) -- 법선 벡터 방법

- **인용**: T. Schuster, J. Ruoff, N. Kerwien, S. Rafler, and W. Osten, "Normal vector method for convergence improvement using the RCWA for crossed gratings," *J. Opt. Soc. Am. A*, vol. 24, no. 9, pp. 2880--2890, 2007. DOI: [10.1364/JOSAA.24.002880](https://doi.org/10.1364/JOSAA.24.002880)
- **요약**: 교차 격자에서 RCWA 수렴성을 개선하기 위한 법선 벡터 방법(Normal Vector Method)을 제안한 논문이다. 재료 경계의 법선 벡터를 활용하여 전기장 연속 조건을 정확하게 적용함으로써, Li 규칙만으로는 불충분한 복잡한 2D 패턴에서도 빠른 수렴을 달성한다. COMPASS의 `fourier_factorization: "normal_vector"` 옵션의 이론적 근거이다.
- **태그**: [RCWA] [수치안정성] [수렴성]

### 5.2 Kim & Lee (2023) -- 고유값 브로드닝 기법

- **인용**: S. Kim and D. Lee, "Eigenvalue broadening technique for stable RCWA simulation of high-contrast gratings," *Comput. Phys. Commun.*, vol. 282, 108547, 2023. DOI: [10.1016/j.cpc.2022.108547](https://doi.org/10.1016/j.cpc.2022.108547)
- **요약**: 고대비 격자의 RCWA 시뮬레이션에서 발생하는 축퇴 고유값 문제에 대한 브로드닝 기법을 제안한 논문이다. 근접한 고유값 쌍을 감지하고 고유벡터를 재직교화하는 후처리 방법으로, COMPASS의 `EigenvalueStabilizer.fix_degenerate_eigenvalues` 메서드가 이 기법을 구현한다. `eigenvalue_broadening: 1e-10` 파라미터의 이론적 근거이다.
- **태그**: [RCWA] [수치안정성] [고유값]

---

## 6. 마이크로렌즈 & 컬러 필터

이미지 센서 광학 구조의 핵심 요소인 마이크로렌즈와 컬러 필터에 관한 논문들이다.

### 6.1 Macleod (2017) -- 박막 광학 필터

- **인용**: H. A. Macleod, *Thin-Film Optical Filters*, 5th ed. Boca Raton, FL: CRC Press, 2017. ISBN: 978-1138198241
- **요약**: 박막 광학 필터의 설계와 분석에 대한 표준 교과서이다. 반사방지 코팅, 간섭 필터, 유전체 다층막 등의 이론과 설계 방법을 상세히 다룬다. COMPASS에서 BARL(Bottom Anti-Reflection Layer) 설계와 컬러 필터 최적화의 참고 자료이다.
- **태그**: [박막광학] [컬러필터] [BARL]

### 6.2 Born & Wolf (1999) -- 광학 원리

- **인용**: M. Born and E. Wolf, *Principles of Optics*, 7th ed. Cambridge: Cambridge University Press, 1999. ISBN: 978-0521642224
- **요약**: 전자기 이론에 기반한 고전 광학의 표준 교과서이다. 프레넬 방정식, 간섭, 회절, 편광 등 COMPASS의 물리적 기반을 이루는 모든 광학 현상의 이론적 참고 자료이다. 특히 다층 박막의 전달 행렬법(TMM) 유도에 사용된다.
- **태그**: [광학이론] [교과서] [프레넬]

### 6.3 Catrysse et al. (2003) -- 집적 컬러 픽셀

- **인용**: P. B. Catrysse, W. Suh, S. Fan, and M. Peeters, "Integrated color pixels in 0.18-um complementary metal oxide semiconductor technology," *J. Opt. Soc. Am. A*, vol. 20, no. 12, pp. 2293--2306, 2003. DOI: [10.1364/JOSAA.20.002293](https://doi.org/10.1364/JOSAA.20.002293)
- **요약**: 0.18um CMOS 공정으로 제작된 집적 컬러 픽셀의 광학 성능을 FDTD로 분석한 논문이다. 컬러 필터 배열, 마이크로렌즈, 금속 배선 등 실제 픽셀 구조의 전자기 시뮬레이션 방법론을 제시했다. COMPASS의 풀스택 픽셀 시뮬레이션 접근법의 선행 연구이다.
- **태그**: [CIS] [컬러필터] [FDTD]

---

## 7. 역설계 & 최적화

나노포토닉스 역설계와 최적화에 관한 핵심 논문들이다.

### 7.1 Molesky et al. (2018) -- 나노포토닉스 역설계 리뷰

- **인용**: S. Molesky, Z. Lin, A. Y. Piggott, W. Jin, J. Vuckovic, and A. W. Rodriguez, "Inverse design in nanophotonics," *Nat. Photonics*, vol. 12, pp. 659--670, 2018. DOI: [10.1038/s41566-018-0246-9](https://doi.org/10.1038/s41566-018-0246-9)
- **요약**: 나노포토닉스 역설계의 핵심 발전을 체계적으로 리뷰한 논문이다. 비선형, 위상, 근접장, 온칩 광학 등 다양한 응용에서의 역설계 기법을 다루며, 토폴로지 최적화와 수반법(adjoint method)의 이론적 기초를 제공한다. COMPASS의 향후 최적화 기능 개발 방향의 근거가 된다.
- **태그**: [최적화] [역설계] [리뷰]

### 7.2 Hughes et al. (2018) -- 수반법 역설계

- **인용**: T. W. Hughes, M. Minkov, I. A. D. Williamson, and S. Fan, "Adjoint method and inverse design for nonlinear nanophotonic devices," *ACS Photonics*, vol. 5, no. 12, pp. 4781--4787, 2018. DOI: [10.1021/acsphotonics.8b01522](https://doi.org/10.1021/acsphotonics.8b01522)
- **요약**: 비선형 나노포토닉 소자의 주파수 영역 역설계를 위한 수반법(adjoint method) 확장을 제시한 논문이다. 비선형 응답을 그래디언트 계산에 직접 포함시키는 기법을 통해 Kerr 비선형 소자의 역설계를 시연했다. COMPASS에서 미분 가능 시뮬레이션 기반 최적화의 이론적 참고 자료이다.
- **태그**: [최적화] [수반법] [비선형]

### 7.3 Piggott et al. (2015) -- 나노포토닉 역설계 시연

- **인용**: A. Y. Piggott, J. Lu, K. G. Lagoudakis, J. Petykiewicz, T. M. Babinec, and J. Vuckovic, "Inverse design and demonstration of a compact and broadband on-chip wavelength demultiplexer," *Nat. Photonics*, vol. 9, pp. 374--377, 2015. DOI: [10.1038/nphoton.2015.69](https://doi.org/10.1038/nphoton.2015.69)
- **요약**: 실리콘 포토닉스 플랫폼에서 콤팩트하고 광대역인 온칩 파장 분리기의 역설계와 실험 시연을 보고한 기념비적 논문이다. 수반법 기반 토폴로지 최적화로 종래 설계 방법으로는 달성할 수 없는 성능을 시연했으며, 나노포토닉 역설계 분야의 초석이 되었다.
- **태그**: [최적화] [역설계] [실험시연]

---

## 8. 미분가능 시뮬레이션

COMPASS에서 사용하는 미분 가능 EM 시뮬레이터 라이브러리의 원 논문들이다.

### 8.1 Jin et al. (2020) -- grcwa (자동미분 가능 RCWA)

- **인용**: W. Jin, W. Li, M. Orenstein, and S. Fan, "Inverse design of lightweight broadband reflector for relativistic lightsail propulsion," *ACS Photonics*, vol. 7, no. 9, pp. 2350--2355, 2020. DOI: [10.1021/acsphotonics.0c00768](https://doi.org/10.1021/acsphotonics.0c00768)
- **요약**: 상대론적 광범위 반사기의 역설계를 위해 자동미분(autograd) 지원 RCWA 구현체인 grcwa를 개발하고 적용한 논문이다. 수반법 그래디언트 계산을 자동미분으로 대체하여 역설계 파이프라인을 단순화했다. COMPASS의 grcwa 솔버 래퍼의 원천 라이브러리이다.
- **태그**: [미분가능시뮬레이션] [RCWA] [grcwa] [최적화]

### 8.2 Kim & Lee (2023) -- torcwa (GPU 가속 RCWA)

- **인용**: C. Kim and B. Lee, "TORCWA: GPU-accelerated Fourier modal method and gradient-based optimization for metasurface design," *Comput. Phys. Commun.*, vol. 282, 108552, 2023. DOI: [10.1016/j.cpc.2022.108552](https://doi.org/10.1016/j.cpc.2022.108552)
- **요약**: PyTorch 기반 GPU 가속 RCWA 시뮬레이터인 torcwa를 소개한 논문이다. GPU 가속을 통해 CPU 대비 큰 속도 향상을 달성하면서 유사한 정확도를 유지하며, PyTorch의 역전파 자동미분을 통한 그래디언트 기반 최적화를 지원한다. COMPASS의 기본 RCWA 솔버이며, 가장 많이 사용되는 백엔드이다.
- **태그**: [미분가능시뮬레이션] [RCWA] [torcwa] [GPU]

### 8.3 Kim et al. (2024) -- meent (미분가능 EM 시뮬레이터)

- **인용**: Y. Kim, A. W. Jung, S. Kim, K. Octavian, D. Heo, C. Park, J. Shin, S. Nam, C. Park, J. Park, et al., "Meent: Differentiable electromagnetic simulator for machine learning," *arXiv preprint*, arXiv:2406.12904, 2024. [arXiv:2406.12904](https://arxiv.org/abs/2406.12904)
- **요약**: 머신러닝과 EM 시뮬레이션의 통합을 위해 개발된 미분 가능 RCWA 솔버인 meent를 소개한 논문이다. 해석적 고유값 분해를 지원하여 수치 안정성을 향상시키고, 신경 연산자 학습, 강화학습 기반 최적화, 그래디언트 기반 역문제 풀이 등 세 가지 응용을 시연했다. COMPASS의 세 번째 RCWA 백엔드이다.
- **태그**: [미분가능시뮬레이션] [RCWA] [meent] [ML]

---

## 부록: COMPASS 솔버 라이브러리 링크

| 솔버 | 유형 | 저장소 | 라이선스 |
|-------|------|--------|----------|
| torcwa | RCWA (PyTorch) | [github.com/kch3782/torcwa](https://github.com/kch3782/torcwa) | MIT |
| grcwa | RCWA (autograd) | [github.com/weiliangjinca/grcwa](https://github.com/weiliangjinca/grcwa) | MIT |
| meent | RCWA (다중 백엔드) | [github.com/kc-ml2/meent](https://github.com/kc-ml2/meent) | MIT |
| fdtd (flaport) | FDTD (PyTorch) | [github.com/flaport/fdtd](https://github.com/flaport/fdtd) | MIT |

---

## 참고 사항

- DOI 링크는 2024년 시점 기준으로 확인되었다. 일부 DOI는 출판사 변경 등으로 리디렉션될 수 있다.
- arXiv 프리프린트의 경우 추후 학술지 게재 시 DOI가 변경될 수 있다.
- COMPASS 코드 내 인용은 `compass/solvers/rcwa/stability.py` 모듈 독스트링과 `docs/about/references.md`에서도 확인할 수 있다.
