"""
MPCCSolve(mod::MPCCmod.MPCC)
"""

module MPCCSolvemod

import MPCCmod.MPCC, MPCCmod.viol_comp,MPCCmod.viol_cons

import SolveRelaxSubProblem.SolveSubproblemAlas

import ParamSetmod.ParamSet

import AlgoSetmod.AlgoSet

import StoppingMPCCmod.StoppingMPCC, StoppingMPCCmod.start,StoppingMPCCmod.stop

import OutputRelaxationmod.OutputRelaxation
import OutputRelaxationmod.Print,OutputRelaxationmod.UpdateFinalOR

type MPCCSolve

 mod::MPCC
 solve_sub_pb::Function
 name_relax::AbstractString

 xj::Vector #itéré courant

 algoset::AlgoSet
 paramset::ParamSet

end

function MPCCSolve(mod::MPCC,x::Vector)

 solve_sub_pb=SolveSubproblemAlas

 #solve_sub_pb=SolveRelaxSubProblem.SolveSubproblemIpOpt # ne marche pas (MPCCtoRelaxNLP problème)
 name_relax="KS" #important que avec IpOpt

 return MPCCSolve(mod,solve_sub_pb,
                  name_relax,x,
                  AlgoSet(),
                  ParamSet(mod.nbc))
end

"""
Accesseur : modifie le point initial
"""

function addInitialPoint(mod::MPCCSolve,x0::Vector)

 mod.xj=x0

 return mod
end

###################################################################################
#
# MAIN FUNCTION
#
###################################################################################

include("mpcc_solve.jl")



#end of module
end
