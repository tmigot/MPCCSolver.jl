gc()

#Tests des packages
include("../ALAS/include.jl")

precision=1e-15

using NLPModels
#package pour plot
using PyPlot
#package pour plot
#using Gadfly

# test : 0 pour tout tester, sinon sélectionner entre 1 et 7.
test=0
done=false

	(test==0 || test==1) ? include("run_test_ThetaFct.jl") : print("")

	(test==0 || test==2) ? include("run_test_Relaxation.jl") : print("")

	# Un exemple simple pour tester :
	 f(x)=x[1]-x[2]
	 G(x)=x[1]
	 H(x)=x[2]
	 c(x)=[1-x[2]]
	 lcon=zeros(1)

	done && (test==0 || test==3) ? include("run_test_ActifMPCCmod.jl") : print("")

	done && (test==0 || test==4) ? include("run_test_UnconstrainedMPCCActif.jl") : print("")

	done && (test==0 || test==5) ? include("run_test_ALASMPCCmod.jl") : print("")

	done && (test==0 || test==6) ? include("run_test_MPCCmod.jl") : print("")

	done && (test==0 || test==7) ? include("run_test_MPCCsolve.jl") : print("")
#FIN des tests
