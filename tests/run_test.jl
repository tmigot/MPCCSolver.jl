gc()
precision=1e-15

using NLPModels
#package pour plot
using PyPlot
#package pour plot
#using Gadfly

# test : 0 pour tout tester, sinon sélectionner entre 1 et 6.
test=0

println("On teste le package Thetamod")
(test==0 || test==1) ? include("run_test_Thetamod.jl") :

println("On teste le package Relaxationmod")
(test==0 || test==2) ? include("run_test_Relaxationmod.jl") :

# Un exemple simple pour tester :
 f(x)=x[1]-x[2]
 G(x)=x[1]
 H(x)=x[2]
 c(x)=[1-x[2]]
 lcon=zeros(1)

println("On teste le module ActifMPCCmod")
(test==0 || test==4) ? include("run_test_ActifMPCCmod.jl") :
#Est-ce que bar_w sert quelque part ?

println("On teste le module UnconstrainedMPCCActif")
(test==0 || test==5) ? include("run_test_UnconstrainedMPCCActif.jl") :

println("On teste le package PASMPCC")
(test==0 || test==6) ? include("run_test_PASMPCC.jl") :

println("On teste le module MPCCmod")
(test==0 || test==3) ? include("run_test_MPCCmod.jl") :

println("Fin des tests")
