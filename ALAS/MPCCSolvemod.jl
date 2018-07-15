"""
MPCCSolve(mod::MPCCmod.MPCC)
"""

module MPCCSolvemod

using MPCCmod
using SolveRelaxSubProblem
using ParamSetmod
using AlgoSetmod

#TO DO:
# - tous les valeurs relatives à la méthode de résolution devraient apparaitre ici pas dans mod:
# - dans l'idéal on aimerait aussi: f(xj), feas(xj), dualfeas(xj), \lambda, or

type MPCCSolve

 mod::MPCCmod.MPCC
 solve_sub_pb::Function
 name_relax::AbstractString

 xj::Vector #itéré courant

 algoset::AlgoSetmod.AlgoSet
 paramset::ParamSetmod.ParamSet

end

function MPCCSolve(mod::MPCCmod.MPCC,x::Vector)

 solve_sub_pb=SolveRelaxSubProblem.SolveSubproblemAlas

 #solve_sub_pb=SolveRelaxSubProblem.SolveSubproblemIpOpt # ne marche pas (MPCCtoRelaxNLP problème)
 name_relax="KS" #important que avec IpOpt

 return MPCCSolve(mod,solve_sub_pb,name_relax,x,AlgoSetmod.AlgoSet(),ParamSetmod.ParamSet(mod.nbc))
end

"""
Accesseur : modifie le point initial
"""

function addInitialPoint(mod::MPCCSolve,x0::Vector)

 mod.xj=x0

 return mod
end


#end of module
end
