module SolveRelaxSubProblem

import MPCCmod: MPCC, consG, MPCCmod.consH
import RlxMPCCSolvemod: RlxMPCCSolve, rlx_solve!, set_x
import RMPCCmod: RMPCC
import RRelaxmod: relax_start!, relax_update!, cons
import RPenmod: RPen

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

 n     = rlx.nlp.n
 ncc   = rlx.nlp.ncc
 mod   = rlx.nlp.mod
 r,s,t = rlx.nlp.r, rlx.nlp.s, rlx.nlp.t
 tb    = rlx.nlp.tb

 #update the initial point
 xk0 = vcat(x0, consG(mod, x0), consH(mod, x0))
 set_x(rlx, xk0)

 #solve the sub-problem
 xk, stat, rpen, oa = rlx_solve!(rlx)

 #update rrelax with the output result rpen
 relax_update!(rlx.rrelax, mod, r, s, t, tb, xk, rpen)

 return xk[1:n], rlx, oa
end

using Ipopt
using MathProgBase

############################################################################
#
# SolveSubproblemIpOpt
#
############################################################################

function solve_subproblem_IpOpt(rlx        :: RlxMPCCSolve,
                                rmpcc      :: RMPCC,
                                name_relax :: AbstractString,
                                x0         :: Vector)

throw(error("This function does not work"))

 n     = rlx.nlp.n
 ncc   = rlx.nlp.ncc
 mod   = rlx.nlp.mod
 r,s,t = rlx.nlp.r, rlx.nlp.s, rlx.nlp.t
 tb    = rlx.nlp.tb

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

 rlx.spas.sub_pb_solved = solved

 return xk, rlx, output
end

#end of module
end
