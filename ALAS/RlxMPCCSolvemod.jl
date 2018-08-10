module RlxMPCCSolvemod

using NLPModels

import AlgoSetmod.AlgoSet
import ParamSetmod.ParamSet

###################################################################################
import ActifMPCCmod.ActifMPCC
import ActifMPCCmod.lsq_computation_multiplier_bool
import ActifMPCCmod.setbeta, ActifMPCCmod.setcrho #pas sûr que ça reste
###################################################################################

import PenMPCCSolvemod.PenMPCCSolve
import PenMPCCSolvemod.grad
import PenMPCCSolvemod.lsq_computation_multiplier_bool
import PenMPCCSolvemod.pen_solve

import RPenmod.RPen
import RPenmod.pen_start!, RPenmod.pen_update!, RPenmod.pen_rho_update!

import PenMPCCmod.PenMPCC,PenMPCCmod.computation_multiplier_bool, PenMPCCmod.jac

import OutputALASmod.OutputALAS

import MPCCmod.MPCC, MPCCmod.viol_contrainte
import MPCCmod.obj, MPCCmod.grad
import MPCCmod.consG, MPCCmod.consH

import StoppingPenmod.StoppingPen

import StoppingRelax.TStoppingPAS
import StoppingRelax.pas_start!,  StoppingRelax.pas_rhoupdate!
import StoppingRelax.pas_stop!,   StoppingRelax.ending_test!

import RRelaxmod.RRelax, RRelaxmod.relax_start!

import Relaxation.psi

import RlxMPCCmod.RlxMPCC

"""
Type ALASMPCC : 

liste des constructeurs :
+RlxMPCCSolve(mod::MPCCmod.MPCC,
              r::Float64,s::Float64,t::Float64,
              prec::Float64, rho::Vector)

liste des méthodes :


liste des accesseurs :

liste des fonctions :
+solvePAS(rlx::RlxMPCCSolve)

"""
#Remark: (mod,r,s,t,tb) represents the relaxed problem.

type RlxMPCCSolve

 nlp      :: RlxMPCC

 #paramètres algorithmiques
 prec     :: Float64 #precision à 0
 rho_init :: Vector #nombre >= 0

 xj       :: Vector

 paramset :: ParamSet
 algoset  :: AlgoSet

 spas     :: TStoppingPAS # stopping the penalization-active-set strategy

 rrelax   :: RRelax

end

function RlxMPCCSolve(mod      :: MPCC,
                      r        :: Float64,
                      s        :: Float64,
                      t        :: Float64,
                      prec     :: Float64,
                      rho      :: Vector,
                      paramset :: ParamSet,
                      algoset  :: AlgoSet,
                      x        :: Vector)

 rho_init = rho
 tb = paramset.tb(r, s, t)

 spas = TStoppingPAS(max_iter  = paramset.ite_max_alas,
                     atol      = prec,
                     goal_viol = paramset.goal_viol,
                     rho_max   = paramset.rho_max)

 rrelax = RRelax(x)

 nlp = RlxMPCC(mod,r,s,t,tb,mod.ncc,mod.n, x = x)

 return RlxMPCCSolve(nlp,
                     prec,rho_init,x,paramset,algoset,spas,rrelax)
end
"""
Accesseur : modifie le point initial
"""

function set_x(rlx  :: RlxMPCCSolve,
               x0   :: Vector)

 rlx.xj = x0

 return rlx
end

############################################################################
#
# Update an RlxMPCCSolve
#
############################################################################
function update_rlx!(rlx  :: RlxMPCCSolve,
                     r    :: Float64,
                     s    :: Float64,
                     t    :: Float64,
                     tb   :: Float64,
                     prec :: Float64,
                     rho  :: Vector)

 rlx.prec = prec
 rlx.rho_init = rho

 rlx.nlp.r  = r
 rlx.nlp.s  = s
 rlx.nlp.t  = t
 rlx.nlp.tb = tb
 #rlx.nlp.tb = rlx.paramset.tb(r, s, t)

 rlx.spas.atol = rlx.prec

 return rlx
end

############################################################################
#solvePAS() : méthode de penalisation/activation de contraintes DHKM 16'
#
#Méthode avec gestion à l'intérieur du paramètre de pénalité
############################################################################

include("rlx_solve.jl")

############################################################################
#
# ...
#
############################################################################

function _update_penalty!(rlx    :: RlxMPCCSolve,
                          ps      :: PenMPCCSolve, 
                          xjk     :: Vector, 
                          UPDATE  :: Bool, 
                          verbose :: Bool)

  if UPDATE

   ps.pen.ρ, ps  = _check_update_rho!(rlx, ps, xjk, verbose)

  end

  #Mise à jour des paramètres de la pénalité Lagrangienne
  ps.pen.u = _lagrange_update(rlx, ps.pen.ρ, xjk, ps.pen.u)

  #met à jour la fonction objectif après la mise à jour des multiplicateurs
  ps.pen.nlp, ps.rpen.fx, ps.rpen.gx = _update_penaltynlp(rlx, ps.pen.ρ, xjk,
                                                          ps.pen.u,
                                                          ps.pen.nlp,
                                                          objpen = ps.rpen.fx,
                                                          gradpen = ps.rpen.gx)

  ps.rpen.lambda, rlx.spas.l_negative = lsq_computation_multiplier_bool(ps, xjk) 
  #ne fait pas tout à fait la même chose
  #ps.rpen.lambda, rlx.spas.l_negative = computation_multiplier_bool(ps.pen,ps.rpen.gx,xjk)

  pen_rho_update!(ps.pen, ps.rpen, xjk)

 return rlx, ps
end

############################################################################
#
# Mise à jour de rho
#
############################################################################

function _check_update_rho!(rlx    :: RlxMPCCSolve,
                           ps      :: PenMPCCSolve,
                           xjkl    :: Vector,
                           verbose :: Bool)

 ht, gradpen = ps.rpen.fx, ps.rpen.gx
 feas = rlx.spas.feasibility

 ρ = ps.pen.ρ
 u = ps.pen.u

 cx = viol_contrainte(rlx.nlp.mod, xjkl)
 ρ = _rho_update(rlx, ρ, ps.crho, abs.(cx))

 #on change le problème donc on réinitialise beta
 #setcrho(ma, rlx.algoset.crho_update(feas, ρ))
 ps.crho = rlx.algoset.crho_update(feas, ρ)

 #met à jour la fonction objectif après la mise à jour de rho
 ps.pen.nlp, ht, gradpen = _update_penaltynlp(rlx, ρ, xjkl, u,
                                              ps.pen.nlp,
                                              objpen=ht,
                                              gradpen=gradpen,
                                              crho=ps.crho)

 #met à jour les différents paramètres horizontaux
 #setbeta(ma,0.0) #on change le problème donc on réinitialise beta
 ps.beta = 0.0
 #ActifMPCCmod.sethess(ma,H) #on change le problème donc on réinitialise Hess

 ps.rpen.fx, ps.rpen.gx = ht, gradpen

 return ρ,ps
end

############################################################################
#Mise à jour de rho:
#function RhoUpdate(rlx  :: RlxMPCCSolve,
#                   ρ    :: Vector,
#                   crho :: Float64,
#                   err  :: Vector;
#                   l    :: Int64=0)
#
# output : Vector
############################################################################

function _rho_update(rlx  :: RlxMPCCSolve,
                     ρ    :: Vector,
                     crho :: Float64,
                     err  :: Vector;
                     l    :: Int64=0)

 nr = length(ρ)

 ρ = min.(ρ*rlx.paramset.rho_update, ones(nr)*rlx.paramset.rho_max)

 return ρ
end

############################################################################
#
# Create/Update the Penalized NLP after an update in ρ
#
# _create_penaltynlp(rlx  :: RlxMPCCSolve,
#                    xj::Vector,ρ::Vector,
#                    usg::Vector,ush::Vector,
#                    uxl::Vector,uxu::Vector,
#                    ucl::Vector,ucu::Vector)
#
# _update_penaltynlp(rlx  :: RlxMPCCSolve,
#                    ρ::Vector,
#                    xj::Vector,
#                    usg::Vector,ush::Vector,
#                    uxl::Vector,uxu::Vector,
#                    ucl::Vector,ucu::Vector,
#                    pen_nlp::NLPModels.AbstractNLPModel;
#                    gradpen::Vector=[],
#                    objpen::Float64=zeros(0),
#                    crho::Float64=1.0)
#
############################################################################

include("nlp_penalty.jl")

############################################################################
# SlackComplementarityProjection : 
# projete le point rlx.xj sur la contrainte de complémentarité
# avec les slack
# 
#
# SlackComplementarityProjection(rlx  :: RlxMPCCSolve) --> x :: Vector 
#                                                         (n + 2ncc)
#
############################################################################

include("projection_init.jl")

############################################################################
#
# Initialize Lagrange Multipliers of the complementarity constraints
#
############################################################################

function _lagrange_comp_init(rlx  :: RlxMPCCSolve,
                             ρ    :: Vector,
                             xj   :: Vector;
                             c    :: Vector = Float64[])

 n     = rlx.nlp.n
 ncc   = rlx.nlp.ncc
 mod   = rlx.nlp.mod
 r,s,t = rlx.nlp.r, rlx.nlp.s, rlx.nlp.t
 tb    = rlx.nlp.tb

 ρ_eqg, ρ_eqh, ρ_lvar, ρ_uvar, ρ_lcons, ρ_ucon = _rho_detail(mod, ρ)
 
 if c == Float64[]

  psiyg = psi(xj[n+1:n+ncc], r, s, t)
  psiyh = psi(xj[n+1+ncc:n+2*ncc], r, s, t)

  #la fonction phi ne fait pas ça ?
  phi = (xj[n+1:n+ncc]-psiyh).*(xj[n+1+ncc:n+2*ncc]-psiyg)

  slackg = xj[n+1:n+ncc]-tb
  slackh = xj[n+1+ncc:n+2*ncc]-tb

 else

  phi = c[2*ncc+1:3*ncc]
  slackg = c[1:ncc]
  slackh = c[ncc+1:2*ncc]

 end

 lphi = max.(phi,zeros(ncc))
 lg = max.(-ρ_eqg.*(slackg), zeros(ncc))
 lh = max.(-ρ_eqh.*(slackh), zeros(ncc))

 return [lg;lh;lphi]
end


############################################################################
#
# Initialize Lagrange Multipliers of the classical constraints
#
############################################################################

function _lagrange_init(rlx  :: RlxMPCCSolve,
                        ρ    :: Vector,
                        xj   :: Vector;
                        c    :: Vector = Float64[])

 n     = rlx.nlp.n
 ncc   = rlx.nlp.ncc
 mod   = rlx.nlp.mod
 r,s,t = rlx.nlp.r, rlx.nlp.s, rlx.nlp.t
 tb    = rlx.nlp.tb
 ncon  = rlx.nlp.mod.mp.meta.ncon

 ρ_eqg, ρ_eqh, ρ_lvar, ρ_uvar, ρ_lcon, ρ_ucon = _rho_detail(mod, ρ)

 if c == Float64[]

  uxl = max.(ρ_lvar.*(mod.mp.meta.lvar-xj[1:n]), zeros(n))
  uxu = max.(ρ_uvar.*(xj[1:n]-mod.mp.meta.uvar), zeros(n))

  if mod.mp.meta.ncon != 0

   cx = NLPModels.cons(mod.mp, xj[1:n])
   nc = length(mod.mp.meta.y0) #nombre de contraintes

   ucl = max.(ρ_lcon.*(mod.mp.meta.lcon-cx), zeros(nc))
   ucu = max.(ρ_ucon.*(cx-mod.mp.meta.ucon), zeros(nc))

  else

   ucl = []
   ucu = []

  end

  usg = zeros(ncc)
  ush = zeros(ncc)

  u = vcat(usg,ush,uxl,uxu,ucl,ucu)

 else

   u = ρ.*c

 end

 return u
end

############################################################################
#
# Update Lagrange Multipliers of the classical constraints
#
############################################################################

function _lagrange_update(rlx  :: RlxMPCCSolve,
                          ρ    :: Vector,
                          xjk  :: Vector,
                          u    :: Vector)

 n     = rlx.nlp.n
 ncc   = rlx.nlp.ncc
 mod   = rlx.nlp.mod

 ρ_eqg, ρ_eqh, ρ_lvar, ρ_uvar, ρ_lcon, ρ_ucon = _rho_detail(mod, ρ)
 usg,ush,uxl,uxu,ucl,ucu = _rho_detail(mod,u)

 uxl = uxl+max.(ρ_lvar.*(xjk[1:n]-mod.mp.meta.uvar), -uxl)
 uxu = uxu+max.(ρ_uvar.*(mod.mp.meta.lvar-xjk[1:n]), -uxu)

 if mod.mp.meta.ncon != 0

  c = NLPModels.cons(mod.mp, xjk[1:n])
  ucl = ucl+max.(ρ_lcon.*(c-mod.mp.meta.ucon), -ucl)
  ucu = ucu+max.(ρ_ucon.*(mod.mp.meta.lcon-c), -ucu)

 end

 if ncc != 0

  G(x) = consG(mod, x)
  H(x) = consH(mod, x)
  usg = usg+ρ_eqg.*(G(xjk[1:n])-xjk[n+1:n+ncc])
  ush = ush+ρ_eqh.*(H(xjk[1:n])-xjk[n+ncc+1:n+2*ncc])

 end

 return vcat(usg,ush,uxl,uxu,ucl,ucu)
end

############################################################################
#
# Initialize ActifMPCC
#
# InitializePenMPCC(rlx  :: RlxMPCCSolve,
#                   xj::Vector,
#                   ρ::Vector,
#                   u::Vector)
# return: PenMPCC
#
############################################################################

include("init_penmpcc.jl")

############################################################################
#
# Initialize ActifMPCC
#
# InitializeMPCCActif(rlx  :: RlxMPCCSolve,
#                     xj::Vector,
#                     ρ::Vector,
#                     usg::Vector,
#                     ush::Vector,
#                     uxl::Vector,
#                     uxu::Vector,
#                     ucl::Vector,
#                     ucu::Vector)
# return: ActifMPCC
#
############################################################################

include("init_penmpccsolve.jl")

############################################################################
#
# Give the detail of the vector Rho:
# return ρ_eq, ρ_ineq_var, ρ_ineq_cons
#
############################################################################

function _rho_detail(mod :: MPCC,
                     ρ   :: Vector)

 nil   = length(mod.mp.meta.lvar)
 niu   = length(mod.mp.meta.uvar)
 nlcon = length(mod.mp.meta.lcon)
 nucon = length(mod.mp.meta.ucon)
 ncc   = mod.ncc

 return ρ[1:ncc],
        ρ[ncc+1:2*ncc],
        ρ[2*ncc+1:2*ncc+nil],
        ρ[2*ncc+nil+1:2*ncc+nil+niu],
        ρ[2*ncc+nil+niu+1:2*ncc+nil+niu+nlcon],
        ρ[2*ncc+nil+niu+nlcon+1:2*ncc+nil+niu+nlcon+nucon]
end

#end of module
end
