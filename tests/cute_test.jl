gc()

#Tests des packages
include("../ALAS/include.jl")

using MPCCmod
using OutputRelaxationmod

using CUTEst
using NLPModels

problems = CUTEst.select()

nlp = CUTEstModel(problems[4])
print(nlp)

#déclare le mpcc :
@printf("Create an MPCC \n")
exemple_nlp=MPCCmod.MPCC(nlp)

@printf("%i %i %i %i %i %i %i %i \n",nlp.counters.neval_obj,
        nlp.counters.neval_cons,nlp.counters.neval_grad,
        nlp.counters.neval_hess,exemple_nlp.G.counters.neval_cons,
        exemple_nlp.G.counters.neval_jac,exemple_nlp.H.counters.neval_cons
        ,exemple_nlp.H.counters.neval_jac)

@printf("Butterfly method:\n")
#résolution avec ALAS Butterfly
xb,fb,orb = MPCCsolve.solve(exemple_nlp)
@printf("%i %i %i %i %i %i %i %i \n",nlp.counters.neval_obj,
        nlp.counters.neval_cons,nlp.counters.neval_grad,
        nlp.counters.neval_hess,exemple_nlp.G.counters.neval_cons,
        exemple_nlp.G.counters.neval_jac,exemple_nlp.H.counters.neval_cons
        ,exemple_nlp.H.counters.neval_jac)

finalize(nlp)
