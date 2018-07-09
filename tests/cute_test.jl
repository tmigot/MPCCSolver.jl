gc()

#Tests des packages
include("../ALAS/include.jl")

using MPCCmod
using OutputRelaxationmod

using CUTEst
using NLPModels

problems_unconstrained = CUTEst.select(contype="unc")
problems_linconstrained = CUTEst.select(contype="linear",min_var=1, max_var=100)
problems_boundconstrained = CUTEst.select(contype="bounds")
problems_quadconstrained = CUTEst.select(contype="quadratic")
problems_genconstrained = CUTEst.select(contype="general")
#problems_linconstrained[50] NaN ?
nlp = CUTEstModel(problems_linconstrained[1])
print(nlp)

#déclare le mpcc :
@printf("Create an MPCC \n")
@time exemple_nlp=MPCCmod.MPCC(nlp)

@printf("%i %i %i %i %i %i %i %i \n",nlp.counters.neval_obj,
        nlp.counters.neval_cons,nlp.counters.neval_grad,
        nlp.counters.neval_hess,exemple_nlp.G.counters.neval_cons,
        exemple_nlp.G.counters.neval_jac,exemple_nlp.H.counters.neval_cons
        ,exemple_nlp.H.counters.neval_jac)

@printf("Butterfly method:\n")
#résolution avec ALAS Butterfly
@time xb,fb,orb,nb_eval = MPCCsolve.solve(exemple_nlp)
try
 oa=orb.inner_output_alas[1]
 @printf("%s	",orb.solve_message)
end
@show nb_eval

finalize(nlp)
