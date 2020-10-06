using LinearAlgebra, LinearOperators, SparseArrays, Printf, Test

include("/home/tmigot/github/MPCC/src/MPCC.jl")
using Main.MPCC

include("../src/MPCCSolver.jl")
using Main.MPCCSolver

using JuMP
using NLPModelsJuMP
using Ipopt, NLPModels, NLPModelsIpopt, Stopping

include("problems.jl")
include("rosenbrock.jl")

printstyled("MPCCSolver param tests... ")
include("test-mpccsolvers-param.jl")
printstyled("passed ✓ \n", color = :green)
if false
printstyled("NLMPCC tests for unconstrained nlps... ")
include("test-nlp-nlmpcc.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("NLMPCC tests for constrained nlps...[TODO] ")
#include("test-nlp-nlmpcc.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("NLMPCC tests for mpccs... ")
include("test-mpcc-nlmpcc.jl")
printstyled("passed ✓ \n", color = :green)
end

printstyled("Theta functions tests... ")
include("test_thetaFct.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("Regularization functions tests... ")
include("test_relaxation_map.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("RlxMPCC tests for mpccs...[TODO] ")
include("test-rlxmpcc.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("Regularization method tests... ")
include("test_relaxation.jl")
printstyled("passed ✓ \n", color = :green)
