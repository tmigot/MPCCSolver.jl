module MPCCSolver

using Ipopt, JuMP, LinearAlgebra, MPCC, NLPModels, NLPModelsIpopt, NLPModelsJuMP, Stopping

include("ParamMPCC.jl")

export ParamMPCC

include("fillMstat.jl")

include("direct-nlp.jl")

export solveIpopt

include("regularization_methods/ThetaFct.jl")
include("regularization_methods/Relaxation.jl")
include("regularization_methods/RlxMPCC.jl")
include("regularization_methods/relaxation_algo.jl")

export mpcc_solve

#export RlxMPCC

end # module
