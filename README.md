# MPCCSolver [Package in Dev]

[![Build Status](https://travis-ci.org/tmigot/MPCCSolver.jl.svg?branch=master)](https://travis-ci.org/tmigot/MPCCSolver.jl)
[![Coverage Status](https://coveralls.io/repos/tmigot/MPCCSolver.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/tmigot/MPCCSolver.jl?branch=julia-0.7)
[![codecov.io](http://codecov.io/github/tmigot/MPCCSolver.jl/coverage.svg?branch=master)](http://codecov.io/github/tmigot/MPCCSolver.jl?branch=master)
<!--[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://tmigot.github.io/MPCC.jl/dev/)
-->

## How to install
Install required pacakages and test the MPCCSolver package with the Julia package manager:
```julia
pkg> add https://github.com/tmigot/MPCC.jl
pkg> add https://github.com/vepiteski/Stopping.jl
pkg> add https://github.com/tmigot/MPCCsolver.jl
pkg> test MPCCSolver.jl
```

## Purpose

Set of algorithms to solve the mathematical program with complementarity/switching/vanishing constraints.
The package follows the same structure as the [MPCC.jl](https://github.com/tmigot/MPCC.jl) and [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).

The package contains the basic tools to use the [Stopping](https://github.com/vepiteski/Stopping.jl) framework.

## Example

More to come...

## Required packages:

[NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl)

[FastClosures](https://github.com/c42f/FastClosures.jl)

[Stopping](https://github.com/vepiteski/Stopping.jl)

[MPCC](https://github.com/tmigot/MPCC.jl))
