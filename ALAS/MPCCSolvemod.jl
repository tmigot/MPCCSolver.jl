"""
MPCCSolve(mod::MPCCmod.MPCC)
"""

module MPCCSolvemod

using MPCCmod
using SolveRelaxSubProblem

type MPCCSolve
 mod::MPCCmod.MPCC
 solve_sub_pb::Function
 name_relax::AbstractString
end

function MPCCSolve(mod::MPCCmod.MPCC)

 solve_sub_pb=SolveRelaxSubProblem.SolveSubproblemAlas
 #solve_sub_pb=SolveRelaxSubProblem.SolveSubproblemIpOpt # ne marche pas (MPCCtoRelaxNLP problème)
 name_relax="KS" #important que avec IpOpt

 return MPCCSolve(mod,solve_sub_pb,name_relax)
end

#end of module
end
