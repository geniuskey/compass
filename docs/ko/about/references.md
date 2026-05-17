<script setup>
import { referencesKo } from './referencesData'
</script>

# 레퍼런스 맵

이 페이지는 모든 주제별 참고문헌을 대체하는 목록이 아닙니다. theory, guide, simulator 문서의 구체적인 주장과 모델 설명은 각 문서 하단의 레퍼런스 섹션을 기준으로 봐야 합니다. 이 페이지는 COMPASS 전반에서 반복적으로 등장하는 기초 논문, 솔버 방법론, 광학 데이터 출처, 소프트웨어 도구를 한곳에 묶은 읽기 지도입니다.

::: tip 이 페이지를 보는 법
먼저 카드에 정리된 기초 방법론 논문을 보고, 특정 시뮬레이터나 구현 세부사항은 해당 문서 하단의 레퍼런스를 따라가면 됩니다.
:::

별도 맵은 여러 문서에 걸쳐 반복되는 출처를 묶을 때만 의미가 있습니다. 모든 로컬 인용을 다시 나열하는 중복 목록으로 키우지는 않습니다.

<ReferenceInteractiveList :references="referencesKo" />

## 방법론 및 데이터 레퍼런스

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
