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
         "ex9.2.1","ex9.2.2","ex9.2.3","ex9.2.4","ex9.2.5",
         "ex9.2.6","ex9.2.7","ex9.2.8","ex9.2.9","flp2","gauvin","hakonsen",
         "hs044-i","jr1","jr2","kth1","kth2","kth3","monteiro","monteiroB",
         "outrata31","outrata32","outrata33","outrata34","qpec1","qpec2",
         "ralph1","ralph2","scholtes1","scholtes2","scholtes3","scholtes4",
         "scholtes5","sl1","stackelberg1"]
fmpec=[]

#10 juillet 17 ++
#good:60,59,57,56,52,49,48,47,46,43,42,41,40,39,36,35,15,14,12, 2, 1
#feas:58,55,54,51,33,32,31,30,29,28,26,25,23,19,17,16,13,11, 8, 7, 5
#nbdd:53, 3
# bad:50,45,44,38,37,34,27,24,22,21,20,18,10, 9, 6, 4
#to check the feasibility 34,27

npb=length(name_pb);npb=10

#output variables:
output=-ones(npb,1)
T=-ones(npb,1)

for i=1:npb

 AMnlp = AmplModel(string("benchmark/macmpec/",name_pb[i],".nl"))

 nlp,G,H=AMtoMPCC(AMnlp)

 @printf("MacMPEC%i: %s \n",i,AMnlp.meta.name)

 #resolution avec ALAS
 exemple_nlp=MPCCmod.MPCC(nlp,G,H)

 @time xb,fb,orb,nb_eval = MPCCsolve.solve(exemple_nlp)
 @printf("%s ",orb.solve_message)
 println(nb_eval)

 T[i,1]=orb.solve_message=="Success"?sum(nb_eval):Inf

 #manque les unbounded
 if orb.solve_message=="Success"
  output[i,1]=0
 elseif orb.solve_message=="Success (with sub-pb failure)"
  output[i,1]=0
 elseif orb.solve_message=="Infeasible"
  output[i,1]=2
 elseif orb.solve_message=="Feasible, but not optimal"
  output[i,1]=1
 else
  output[i,1]=3
 end

 amplmodel_finalize(AMnlp)
end
