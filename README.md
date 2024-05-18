# PhotonSurrogateModel

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chrhck.github.io/PhotonSurrogateModel.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://chrhck.github.io/PhotonSurrogateModel.jl/dev/)
[![Build Status](https://github.com/chrhck/PhotonSurrogateModel.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/chrhck/PhotonSurrogateModel.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/chrhck/PhotonSurrogateModel.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/chrhck/PhotonSurrogateModel.jl)

This package provides a surrogate model for modelling photon arrival time distributions for neutrino telescopes.
For an example see `examples/plot_surrogate_model.ipynb`.

## Installation

This package is registered in the PLEnuM julia package [registry](https://github.com/PLEnuM-group/julia-registry). In order to use this registry run:
```{julia}
using Pkg
pkg"registry add https://github.com/PLEnuM-group/julia-registry"
```

Then install the package:
```{julia}
using Pkg
pkg"add PhotonSurrogateModel"
```
