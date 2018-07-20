"""
MPCCSolve(mod::MPCCmod.MPCC)
"""

module MPCCSolvemod

import MPCCmod.MPCC, MPCCmod.viol_comp

import SolveRelaxSubProblem.SolveSubproblemAlas

import ParamSetmod.ParamSet
import ParamMPCCmod.ParamMPCC
import AlgoSetmod.AlgoSet

import StoppingMPCCmod.StoppingMPCC, StoppingMPCCmod.start,StoppingMPCCmod.stop

import OutputRelaxationmod.OutputRelaxation
import OutputRelaxationmod.Print,OutputRelaxationmod.UpdateFinalOR




type MPCCSolve

 mod::MPCC
 name_relax::AbstractString

 xj::Vector #itéré courant

 algoset::AlgoSet
 paramset::ParamSet
 parammpcc :: ParamMPCC

end

function MPCCSolve(mod::MPCC,x::Vector)

 algoset = AlgoSet()

 #solve_sub_pb=SolveRelaxSubProblem.SolveSubproblemIpOpt # ne marche pas (MPCCtoRelaxNLP problème)
 name_relax="KS" #important que avec IpOpt

 return MPCCSolve(mod,
                  name_relax,x,
                  AlgoSet(),
                  ParamSet(mod.nbc),
                  ParamMPCC(mod.nbc))
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
