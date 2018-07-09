#Functions that solves the non-linear relaxed program.
#Sélectionne et résoud le problème d'optimisation dans le sous domaine actif.
export RelaxedNLsolve

include("RelaxedNLPenalization.jl")

using ALASMPCCmod

function RelaxedNLsolve(alas::ALASMPCCmod.ALASMPCC,x0::Vector,feas::Float64,gradpen::Vector,method::Function)
 return method(alas::ALASMPCCmod.ALASMPCC,x0::Vector,feas::Float64,gradpen::Vector)
end
