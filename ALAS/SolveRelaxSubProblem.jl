"""
Package de fonctions pour définir la précision
sur la réalisabilité dual pendant la pénalisation
'sans contraintes'.


liste des fonctions :
SolveSubproblemAlas(mod::MPCCmod.MPCC,
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
# - Ca serait plus économique d'avoir une structure ss-pb pour toute la boucle 
#   de relaxation plutôt qu'en créer un à chaque fois.

module SolveRelaxSubProblem

import MPCCmod.MPCC
import MPCCmod.consG, MPCCmod.consH

import ALASMPCCmod.ALASMPCC, ALASMPCCmod.solvePAS
import AlgoSetmod.AlgoSet
import ParamSetmod.ParamSet

import RMPCCmod.RMPCC, RMPCCmod.update!
import RRelaxmod.relax_start!, RRelaxmod.relax_update!

"""
Methode pour résoudre le sous-problème relaxé :
"""
function SolveSubproblemAlas(mod        :: MPCC,
                             rmpcc      :: RMPCC,
                             r          :: Float64,
                             s          :: Float64,
                             t          :: Float64,
                             ρ          :: Vector,
                             name_relax :: AbstractString,
                             paramset   :: ParamSet,
                             algoset    :: AlgoSet,
                             x0         :: Vector,
                             prec       :: Float64)

############################################################
 #Initialize/Update the sub-problem
 x0=vcat(x0,consG(mod,x0),consH(mod,x0))
 alas = ALASMPCC(mod,r,s,t,prec,ρ,paramset,algoset,x0)

 #initialize rrelax with what we know from rmpcc
 relax_update!(alas.rrelax, mod, r,s,t,alas.tb,x0, 
               fx = rmpcc.fx, gx = rmpcc.gx, feas = rmpcc.feas)
 relax_start!(alas.rrelax,alas.mod,alas.r,alas.s,alas.t,alas.tb,x0)
############################################################

 #solve the sub-problem
 xk,stat,ρ,oa = solvePAS(alas) #lg,lh,lphi,s_xtab dans le output

 #update the result with the output of solve
 update!(rmpcc, mod, xk[1:mod.n])

 return xk,rmpcc,stat==0,ρ,oa
end

using Ipopt
using MathProgBase

"""
Methode pour résoudre le sous-problème relaxé :
"""
function SolveSubproblemIpOpt(mod        :: MPCC,
                              rmpcc      :: RMPCC,
                              r          :: Float64,
                              s          :: Float64,
                              t          :: Float64,
                              rho        :: Vector,
                              name_relax :: AbstractString,
                              paramset   :: ParamSet,
                              algoset    :: AlgoSet,
                              x0         :: Vector,
                              prec       :: Float64)
 solved=true
 #nlp_relax = MPCCtoRelaxNLP(mod,r,s,t,name_relax) #si nb_comp>0
 nlp_relax=mod.mp
 output=[]

 # resolution du sous-problème avec IpOpt
 model_relax = NLPModels.NLPtoMPB(nlp_relax, IpoptSolver(print_level=0,tol=paramset.precmpcc))
 MathProgBase.optimize!(model_relax)

 if MathProgBase.status(model_relax) == :Optimal
  xk = MathProgBase.getsolution(model_relax)
 else
  xk=x0
  solved=false
 end

 update!(rmpcc, mod, xk[1:mod.n])

 return xk,rmpcc,solved,rho,output
end

###
#
# !!! Ne marche pas comme ça !!!
#
###
include("mpcc_to_relax_nlp.jl")

#end of module
end
