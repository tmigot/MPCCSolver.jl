#fist we check if all package are installed:
package_check=false

if package_check
#JuMP
#NLPModels
#CUTEst
#Pkg.add("BenchmarkProfiles")
end

#include file
include("../ALAS/OutputLSmod.jl")
include("../ALAS/OutputALASmod.jl")
include("../ALAS/OutputRelaxationmod.jl")

include("../ALAS/Thetafunc.jl")
include("../ALAS/Relaxation.jl")
include("../ALAS/Penalty.jl")
include("../ALAS/ParamSetmod.jl")
include("../ALAS/ActifMPCCmod.jl")
include("../ALAS/DDirection.jl")
include("../ALAS/LineSearch.jl")
include("../ALAS/AlgoSetmod.jl")
include("../ALAS/UnconstrainedMPCCActif.jl")
include("../ALAS/MPCCmod.jl")
include("../ALAS/ALASMPCCmod.jl")
include("../ALAS/MPCCsolve.jl")
include("../ALAS/MPCC2DPlot.jl")
