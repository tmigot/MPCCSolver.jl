#Tests des packages
include("../ALAS/include.jl")

using MPCCmod
using OutputRelaxationmod

using OptimizationProblems
using MathProgBase
using NLPModels
#8
sprob=names(OptimizationProblems)[91]
#sprob=:hs1
@time n=4
nlp=AbstractNLPModel
try
nlp = MathProgNLPModel(eval(sprob)(n))
catch
nlp = MathProgNLPModel(eval(sprob)())
end
nlp.meta.ncon==0 && @printf("Unconstrained problem? yes!\n")

#déclare le mpcc :
println("Create an MPCC")
@time exemple_nlp=MPCCmod.MPCC(nlp)
@printf("%i %i %i %i %i %i %i %i \n",nlp.counters.neval_obj,
        nlp.counters.neval_cons,nlp.counters.neval_grad,
        nlp.counters.neval_hess,exemple_nlp.G.counters.neval_cons,
        exemple_nlp.G.counters.neval_jac,exemple_nlp.H.counters.neval_cons
        ,exemple_nlp.H.counters.neval_jac)

println("Butterfly method:")
#résolution avec ALAS Butterfly
@time xb,fb,orb = MPCCsolve.solve(exemple_nlp)
@printf("%i %i %i %i %i %i %i %i \n",nlp.counters.neval_obj,
        nlp.counters.neval_cons,nlp.counters.neval_grad,
        nlp.counters.neval_hess,exemple_nlp.G.counters.neval_cons,
        exemple_nlp.G.counters.neval_jac,exemple_nlp.H.counters.neval_cons
        ,exemple_nlp.H.counters.neval_jac)
