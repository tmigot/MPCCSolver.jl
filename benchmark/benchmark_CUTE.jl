#Tests des packages
include("../ALAS/include.jl")

using MPCCmod
using OutputRelaxationmod
using Penalty
using DDirection
using LineSearch
using AlgoSetmod

using CUTEst
using NLPModels

using BenchmarkProfiles

problems_unconstrained = CUTEst.select(contype="unc")
problems_linconstrained = CUTEst.select(contype="linear",min_var=1, max_var=10)
problems_boundconstrained = CUTEst.select(contype="bounds")
problems_quadconstrained = CUTEst.select(contype="quadratic")
problems_genconstrained = CUTEst.select(contype="general")
problems=CUTEst.select(min_var=1, max_var=10)
m=length(problems)

#USER Choice :

pb=problems
n=length(pb)

algo1=AlgoSetmod.AlgoSet(Penalty.Quadratic,DDirection.CGHZ,LineSearch.ArmijoWolfe)
algo2=AlgoSetmod.AlgoSet(Penalty.Quadratic,DDirection.CGHZ,LineSearch.ArmijoWolfeHZ)
algo3=AlgoSetmod.AlgoSet(Penalty.Quadratic,DDirection.CGHZ,LineSearch.Armijo)

#....

T=zeros(n,3)

@printf("Start of a benchmark test from CUTEst with %i problems.\n",n)

for i=1:min(n,m)
 nlp = CUTEstModel(pb[i])
 @printf("CUTEst %i: %s \n",i,nlp.meta.name)

 #resolution avec ALAS
 exemple_nlp=MPCCmod.MPCC(nlp,algo1)
 xb,fb,orb,nb_eval = MPCCsolve.solve(exemple_nlp)
 @printf("%s ",orb.solve_message)
 println(nb_eval)
 T[i,1]=orb.solve_message=="Success"?sum(nb_eval):Inf

 exemple_nlp=MPCCmod.MPCC(nlp,algo2)
 xb,fb,orb,nb_eval = MPCCsolve.solve(exemple_nlp)
 @printf("%s ",orb.solve_message)
 println(nb_eval)
 T[i,2]=orb.solve_message=="Success"?sum(nb_eval):Inf

 exemple_nlp=MPCCmod.MPCC(nlp,algo3)
 xb,fb,orb,nb_eval = MPCCsolve.solve(exemple_nlp)
 @printf("%s ",orb.solve_message)
 println(nb_eval)
 T[i,3]=orb.solve_message=="Success"?sum(nb_eval):Inf

 finalize(nlp)
end

performance_profile(T,["Solver 1","Sovler 2","Sovler 3"], title="Celebrity Death Match")
