#Tests des packages
include("../ALAS/include.jl")

using MPCCmod
using OutputRelaxationmod

using OptimizationProblems
using MathProgBase
using NLPModels
#8
sprob=names(OptimizationProblems)[32]
println(sprob)
sprob=:woods
n=5

nlp = MathProgNLPModel(eval(sprob)(n))
nlpt = SimpleNLPModel(x->obj(nlp,x), nlp.meta.x0, g=x->ones(length(x)), Hp=(x,v)->ones(length(x),length(x)))

nlp.meta.ncon==0 && println("Unconstrained problem? yes!")

#déclare le mpcc :
println("Create an MPCC")
exemple_nlp=MPCCmod.MPCC(nlp)
println(nlp.counters.neval_obj)
println(nlp.counters.neval_cons)
println(nlp.counters.neval_grad)
println(nlp.counters.neval_hess)
println(exemple_nlp.G.counters.neval_cons)
println(exemple_nlp.G.counters.neval_jac)
println(exemple_nlp.H.counters.neval_cons)
println(exemple_nlp.H.counters.neval_jac)

println("Butterfly method:")
#résolution avec ALAS Butterfly
xb,fb,orb = MPCCsolve.solve(exemple_nlp)
println(nlp.counters.neval_obj)
println(nlp.counters.neval_cons)
println(nlp.counters.neval_grad)
println(nlp.counters.neval_hess)
println(exemple_nlp.G.counters.neval_cons)
println(exemple_nlp.G.counters.neval_jac)
println(exemple_nlp.H.counters.neval_cons)
println(exemple_nlp.H.counters.neval_jac)
println("test")
