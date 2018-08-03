#fist we check if all package are installed:
#include("installer.jl")

#include file
include("../ALAS/OutputLSmod.jl")
include("../ALAS/OutputALASmod.jl")

include("../ALAS/Thetafunc.jl")
include("../ALAS/Relaxation.jl")

include("../ALAS/Penalty.jl")
include("../ALAS/ParamSetmod.jl")

include("../ALAS/PenMPCCmod.jl")
include("../ALAS/RPenmod.jl")
include("../ALAS/StoppingPenmod.jl")

include("../ALAS/RActifmod.jl")
include("../ALAS/Stopping.jl")
include("../../LSDescentMethods-master/src/LSDescentMethods.jl")
include("../ALAS/ActifMPCCmod.jl")

include("../ALAS/DDirection.jl")
include("../ALAS/LineSearch.jl")
#include("../ALAS/ScalingDual.jl")
include("../ALAS/CRhoUpdate.jl")
include("../ALAS/AlgoSetmod.jl")
#include("../ALAS/UnconstrainedMPCCActif.jl")

include("../ALAS/MPCCmod.jl") #parce qu'on a pas de type Relax
include("../ALAS/RRelaxmod.jl")
include("../ALAS/StoppingRelax.jl")
include("../ALAS/RlxMPCCSolvemod.jl")

include("../ALAS/RMPCCmod.jl")
include("../ALAS/SolveRelaxSubProblem.jl")
include("../ALAS/ParamMPCCmod.jl")
include("../ALAS/OutputRelaxationmod.jl")
include("../ALAS/StoppingMPCCmod.jl")
include("../ALAS/MPCCSolvemod.jl")
