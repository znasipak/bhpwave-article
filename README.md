# bhpwave-article 
[![LaTex Build](https://github.com/znasipak/bhpwave-article/actions/workflows/compile-release.yml/badge.svg)](https://github.com/znasipak/bhpwave-article/actions/workflows/compile-release.yml/badge.svg) 
[![Read the article](https://img.shields.io/badge/PDF-latest-blue.svg?style=flat)](https://github.com/znasipak/bhpwave-article/raw/gh-action-result/pdflatex/ms.pdf)
[![DOI](https://img.shields.io/badge/arXiv-2310.19706-B31B1B)](https://doi.org/10.48550/arXiv.2310.19706)
[![DOI](https://img.shields.io/badge/PhysRevD-109.044020-purple)]([https://doi.org/10.48550/arXiv.2310.19706](https://doi.org/10.1103/PhysRevD.109.044020))

We present `bhpwave`: a new Python-based, open-source tool for generating the gravitational waveforms of stellar-mass compact objects undergoing quasi-circular inspirals into rotating massive black holes. These binaries, known as extreme-mass-ratio inspirals (EMRIs), are exciting mHz gravitational wave sources for future space-based detectors such as LISA. Relativistic models of EMRI gravitational wave signals are necessary to unlock the full scientific potential of mHz detectors, yet few open-source EMRI waveform models exist. Thus we built `bhpwave`, which uses the adiabatic approximation from black hole perturbation theory to rapidly construct gravitational waveforms based on the leading-order inspiral dynamics of the binary. In this work, we present the theoretical and numerical foundations underpinning `bhpwave`. We also demonstrate how `bhpwave` can be used to assess the impact of EMRI modeling errors on LISA gravitational wave data analysis. In particular, we find that for retrograde orbits and slowly-spinning black holes we can mismodel the gravitational wave phasing by as much as $\sim 10$ radians without significantly biasing EMRI parameter estimation.

Published versions of the paper can also be found on the [arXiv](https://doi.org/10.48550/arXiv.2310.19706) and [Physical Review D](https://doi.org/10.1103/PhysRevD.109.044020).
