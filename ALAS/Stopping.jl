module Stopping

using NLPModels
using JuMP
importall MPCCmod

include("stopping_unconstrained.jl")
include("stopping_pas.jl")

# end of module
end
