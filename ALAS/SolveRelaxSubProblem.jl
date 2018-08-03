"""
Package de fonctions pour définir la précision
sur la réalisabilité dual pendant la pénalisation
'sans contraintes'.


liste des fonctions :
SolveSubproblemrlx(mod::MPCCmod.MPCC,
                    r::Float64,s::Float64,t::Float64,
                    rho::Vector,name_relax::AbstractString)
SolveSubproblemIpopt(mod::MPCCmod.MPCC,
                              r::Float64,s::Float64,t::Float64,
                              rho::Vector,name_relax::AbstractString)

MPCCtoRelaxNLP(mod::MPCC, r::Float64, s::Float64, t::Float64,
              relax::AbstractString)

"""

# TO DO List
#Major :
# - MPCCtoRelaxNLP : bug à corriger

module SolveRelaxSubProblem #est-ce que c'est vraiment un module ?

import MPCCmod.MPCC
import MPCCmod.consG, MPCCmod.consH

import RlxMPCCSolvemod.RlxMPCCSolve, RlxMPCCSolvemod.rlx_solve!, RlxMPCCSolvemod.set_x

import RMPCCmod.RMPCC
import RRelaxmod.relax_start!, RRelaxmod.relax_update!
import RRelaxmod.cons

import RPenmod.RPen

"""
Methode pour résoudre le sous-problème relaxé :
"""
############################################################################
#
# SolveSubproblemrlx
#
############################################################################
#+ économique comme point initial, mais moins bon ?
#xk0= length(rlx.xj) == rlx.mod.n ? vcat(x0,consG(rlx.mod,x0),consH(rlx.mod,x0)) : rlx.xj

function solve_subproblem_rlx(rlx       :: RlxMPCCSolve,
                             rmpcc      :: RMPCC,
                             name_relax :: AbstractString,
                             x0         :: Vector)

 #update the initial point
 xk0 = vcat(x0, consG(rlx.mod, x0), consH(rlx.mod, x0))
 set_x(rlx, xk0)

 #solve the sub-problem
 xk, stat, rpen, oa = rlx_solve!(rlx)

 #update rrelax with the output result rpen
 relax_update!(rlx.rrelax, rlx.mod, rlx.r, rlx.s, rlx.t, rlx.tb, xk, rpen)

 return xk[1:rlx.mod.n], rlx, stat==0, oa
end

using Ipopt
using MathProgBase

"""
Methode pour résoudre le sous-problème relaxé :
"""
############################################################################
#
# SolveSubproblemIpOpt
#
############################################################################

function solve_subproblem_IpOpt(rlx        :: RlxMPCCSolve,
                                rmpcc      :: RMPCC,
                                name_relax :: AbstractString,
                                x0         :: Vector)

 solved = true
 #nlp_relax = MPCCtoRelaxNLP(rlx.mod,rlx.r,rlx.s,rlx.t,name_relax) #si ncc>0
 nlp_relax = mod.mp
 output = []

 # resolution du sous-problème avec IpOpt
 model_relax = NLPModels.NLPtoMPB(nlp_relax, IpoptSolver(print_level = 0, tol = rlx.prec))
 MathProgBase.optimize!(model_relax)

 if MathProgBase.status(model_relax) == :Optimal
  xk = MathProgBase.getsolution(model_relax)
 else
  xk = x0
  solved = false
 end

 return xk, rlx, rmpcc, solved, output
end

############################################################################
#
# !!! Ne marche pas comme ça !!!
#
############################################################################

include("mpcc_to_relax_nlp.jl")

#end of module
end
