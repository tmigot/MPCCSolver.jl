#Tests des packages
include("../ALAS/include.jl")
include("AMtoMPCC.jl")

using MPCCmod
using OutputRelaxationmod
using Penalty
using DDirection
using LineSearch
using AlgoSetmod

using CUTEst
using NLPModels
using AmplNLReader

using BenchmarkProfiles

algo1=AlgoSetmod.AlgoSet(Penalty.Quadratic,DDirection.NwtdirectionSpectral,LineSearch.ArmijoWolfe)

i=1;n=1;T=zeros(n,3);

AMnlp = AmplModel("benchmark/macmpec/scholtes2.nl")

nlp,G,H=AMtoMPCC(AMnlp)

 @printf("MacMPEC%i: %s \n",i,AMnlp.meta.name)

 #resolution avec ALAS
 exemple_nlp=MPCCmod.MPCC(nlp,G,H)
@show exemple_nlp.nb_comp
@time xb,fb,orb,nb_eval = MPCCsolve.solve(exemple_nlp)
 @printf("%s ",orb.solve_message)
 println(nb_eval)
@show orb.solve_message=="Success" sum(nb_eval)
 T[i,1]=orb.solve_message=="Success"?sum(nb_eval):Inf
@show T[i,1]

amplmodel_finalize(AMnlp)
