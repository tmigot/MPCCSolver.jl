#fist we check if all package are installed:
#include("installer.jl")

#include file
include("../ALAS/MPCCMetamod.jl")
include("../ALAS/MPCCmod.jl")

include("../ALAS/OutputLSmod.jl")
include("../ALAS/OutputALASmod.jl")

include("../ALAS/ThetaFct.jl")
include("../ALAS/Relaxation.jl")

include("../ALAS/RlxMPCCmod.jl")

include("../ALAS/Penalty.jl")

include("../ALAS/PenMPCCmod.jl")

include("../ALAS/ParamSetmod.jl")

include("../ALAS/RUncstrndmod.jl")
include("../ALAS/ActifModelmod.jl")
include("../ALAS/Stopping1Dmod.jl")
include("../ALAS/UncstrndSolvemod.jl")

include("../ALAS/Stopping.jl") #changer le nom
include("../ALAS/RPenmod.jl")
include("../ALAS/StoppingPenmod.jl")

include("../ALAS/RActifmod.jl")
include("../../LSDescentMethods-master/src/LSDescentMethods.jl")
include("../ALAS/ActifMPCCmod.jl") #nommer ActifMPCCSolvemod ?

include("../ALAS/DDirection.jl")
include("../ALAS/LineSearch.jl")
#include("../ALAS/ScalingDual.jl")
include("../ALAS/CRhoUpdate.jl")
include("../ALAS/AlgoSetmod.jl")
include("../ALAS/PenMPCCSolvemod.jl")

include("../ALAS/RRelaxmod.jl")
include("../ALAS/StoppingRelax.jl")
include("../ALAS/RlxMPCCSolvemod.jl")

include("../ALAS/RMPCCmod.jl")
include("../ALAS/SolveRelaxSubProblem.jl")
include("../ALAS/ParamMPCCmod.jl")
include("../ALAS/OutputRelaxationmod.jl")
include("../ALAS/StoppingMPCCmod.jl")
include("../ALAS/MPCCSolvemod.jl")
