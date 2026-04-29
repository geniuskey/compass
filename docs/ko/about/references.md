<script setup>
import { referencesKo } from './referencesData'
</script>

# 참고 문헌

COMPASS의 물리학과 알고리즘을 뒷받침하는 주요 논문 및 자료입니다. 기초가 되는 핵심 논문들의 요약과 관련 시각 자료를 보려면 아래 카드를 클릭하세요.

<ReferenceInteractiveList :references="referencesKo" />

## 추가 참고 문헌

### RCWA 이론 및 개선
- M.G. Moharam, D.A. Pommet, E.B. Grann, and T.K. Gaylord, "Stable implementation of the rigorous coupled-wave analysis for surface-relief gratings: enhanced transmittance matrix approach," *J. Opt. Soc. Am. A*, vol. 12, no. 5, pp. 1077-1086, 1995.
- L. Li, "New formulation of the Fourier modal method for crossed surface-relief gratings," *J. Opt. Soc. Am. A*, vol. 14, no. 10, pp. 2758-2767, 1997.
- T. Schuster, J. Ruoff, N. Kerwien, S. Rafler, and W. Osten, "Normal vector method for convergence improvement using the RCWA for crossed gratings," *J. Opt. Soc. Am. A*, vol. 24, no. 9, pp. 2880-2890, 2007.
- S. Kim and D. Lee, "Eigenvalue broadening technique for stable RCWA simulation of high-contrast gratings," *Comput. Phys. Commun.*, vol. 282, 108547, 2023.

### FDTD 방법
- A. Taflove and S.C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed. Artech House, 2005.

### 실리콘 광학 특성
- M.A. Green, "Self-consistent optical parameters of intrinsic silicon at 300 K including temperature coefficients," *Solar Energy Materials and Solar Cells*, vol. 92, no. 11, pp. 1305-1310, 2008.
- E.D. Palik, *Handbook of Optical Constants of Solids*, Academic Press, 1998.

### 이미지 센서 물리학
- S.K. Mendis, S.E. Kemeny, R.C. Gee, B. Pain, C.O. Staller, Q. Kim, and E.R. Fossum, "CMOS active pixel image sensors for highly integrated imaging systems," *IEEE J. Solid-State Circuits*, vol. 32, no. 2, pp. 187-197, 1997.

### 박막 광학
- H.A. Macleod, *Thin-Film Optical Filters*, 5th ed. CRC Press, 2017.
- M. Born and E. Wolf, *Principles of Optics*, 7th ed. Cambridge University Press, 1999.

## 솔버 라이브러리

- **torcwa**: PyTorch 기반 RCWA. GPU 가속 지원.
- **grcwa**: GPU RCWA 구현체.
- **meent**: 해석적 고유값 분해를 사용하는 메타표면(metasurface) 전자기 솔버.
- **fdtd (flaport)**: PyTorch 기반 FDTD.
- **Meep**: MIT 전자기 방정식 전파(MIT Electromagnetic Equation Propagation, 오픈소스 FDTD).

## 소프트웨어 도구

- **PyTorch**: [pytorch.org](https://pytorch.org/) -- GPU 연산 프레임워크
- **Hydra**: [hydra.cc](https://hydra.cc/) -- 설정 관리
- **Pydantic**: [pydantic.dev](https://docs.pydantic.dev/) -- 데이터 유효성 검증
- **VitePress**: [vitepress.dev](https://vitepress.dev/) -- 문서 프레임워크
