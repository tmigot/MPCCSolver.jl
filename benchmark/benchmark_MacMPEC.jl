#Tests des packages
include("../ALAS/include.jl")
include("AMtoMPCC.jl")

using MPCCmod
using OutputRelaxationmod
using Penalty
using DDirection
using LineSearch
using AlgoSetmod

using NLPModels
using AmplNLReader

using BenchmarkProfiles

name_pb=["bard1","bard1m","bard2","bard2m","bard3","bard3m","bilevel1",
         "bilevel1m","bilevel2","bilevel2m","bilevel3","bilin",
         "dempe","desilva","df1","ex9.1.1","ex9.1.2","ex9.1.3","ex9.1.4",
         "ex9.1.5","ex9.1.6","ex9.1.7","ex9.1.8","ex9.1.9","ex9.1.10",
         "ex9.2.1","ex9.2.2","ex9.2.3","ex9.2.4","ex9.2.5","ex9.2.5",
         "ex9.2.6","ex9.2.7","ex9.2.8","ex9.2.9","flp2","gauvin","hakonsen",
         "hs044-i","jr1","jr2","kth1","kth2","kth3","monteiro","monteiroB",
         "outrata31","outrata32","outrata33","outrata34","qpec1","qpec2",
         "ralph1","ralph2","scholtes1","scholtes2","scholtes3","scholtes4",
         "scholtes5","sl1","stackelberg1"];

#good: 1,61,5, 11, 16, 17, 19,20, 21, 23, 25, 26, 29, 30, 31, 32, 33, 35, 37, 44, 51, 52, 59
#good(-): 7, 8 (pb de critère d'arrêt)
#bad: 60, 4, 6, 12, 24, 27, 28, 39
#ubnbd: 2, 3, 34, 54
#??? 9, 10, 13, 18, 22, 43(+), 55(+), 56(+)
#0ité: 13,14, 36, 40, 41, 42, 47, 48, 49, 50, 53, 57, 58
#AMtoMPCC error: 15, 45, 46, 38(AMPL)

i=1;n=1;T=zeros(n,3);
AMnlp = AmplModel(string("benchmark/macmpec/",name_pb[59],".nl"))
#AMnlp = AmplModel("benchmark/macmpec/scholtes2.nl")

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

#amplmodel_finalize(AMnlp);
