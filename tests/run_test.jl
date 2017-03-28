gc()

#Tests des packages
include("../ALAS/Thetafunc.jl")
include("../ALAS/Relaxation.jl")
include("../ALAS/ActifMPCCmod.jl")
include("../ALAS/DDirection.jl")
include("../ALAS/LineSearch.jl")
include("../ALAS/ParamSetmod.jl")
include("../ALAS/AlgoSetmod.jl")
include("../ALAS/UnconstrainedMPCCActif.jl")
include("../ALAS/MPCCmod.jl")
include("../ALAS/ALASMPCCmod.jl")
include("../ALAS/MPCCsolve.jl")
include("../ALAS/MPCC2DPlot.jl")

precision=1e-15

using NLPModels
#package pour plot
using PyPlot
#package pour plot
#using Gadfly

# test : 0 pour tout tester, sinon sélectionner entre 1 et 7.
test=0

	(test==0 || test==1) ? include("run_test_Thetafunc.jl") : include("")

	(test==0 || test==2) ? include("run_test_Relaxation.jl") : include("")

	# Un exemple simple pour tester :
	 f(x)=x[1]-x[2]
	 G(x)=x[1]
	 H(x)=x[2]
	 c(x)=[1-x[2]]
	 lcon=zeros(1)

	(test==0 || test==3) ? include("run_test_ActifMPCCmod.jl") : include("")

	(test==0 || test==4) ? include("run_test_UnconstrainedMPCCActif.jl") : include("")

	(test==0 || test==5) ? include("run_test_ALASMPCCmod.jl") : include("")

	(test==0 || test==6) ? include("run_test_MPCCmod.jl") : include("")

	(test==0 || test==7) ? include("run_test_MPCCsolve.jl") : include("")
#FIN des tests
