#Tests des packages
include("../ALAS/include.jl")

using MPCCmod
using OutputRelaxationmod

using CUTEst
using NLPModels

using BenchmarkProfiles

problems_unconstrained = CUTEst.select(contype="unc")
problems_linconstrained = CUTEst.select(contype="linear")
problems_boundconstrained = CUTEst.select(contype="bounds")
problems_quadconstrained = CUTEst.select(contype="quadratic")
problems_genconstrained = CUTEst.select(contype="general")
problems=CUTEst.select()
n=length(problems)

#USER Choice :

pb=problems_unconstrained
n=5

#....

T=zeros(n,3)

for i=1:min(n,length(pb))
 nlp = CUTEstModel(pb[i])
 @printf("CUTEst %i: %s	",i,nlp.meta.name)

 #resolution avec ALAS
 exemple_nlp=MPCCmod.MPCC(nlp)
 xb,fb,orb,nb_eval = MPCCsolve.solve(exemple_nlp)
 @printf("%s",orb.solve_message)
 println(nb_eval)

 T[i,1]=sum(nb_eval)
 T[i,2]=sum(nb_eval)+2*rand()-1
 T[i,3]=sum(nb_eval)+2*rand()-1

 finalize(nlp)
end

performance_profile(T,["Solver 1","Sovler 2","Sovler 3"], title="Celebrity Death Match")
