module ALASMPCCmod

using NLPModels

import AlgoSetmod.AlgoSet
import ParamSetmod.ParamSet

import ActifMPCCmod.ActifMPCC,ActifMPCCmod.grad
import ActifMPCCmod.ActiveCons #could not import ActifMPCCmod.ActiveCons into ALASMPCCmod
import ActifMPCCmod.LSQComputationMultiplierBool,ActifMPCCmod.RelaxationRule!
import ActifMPCCmod.setbeta, ActifMPCCmod.setcrho #pas sûr que ça reste

import RPenmod.RPen
import PenMPCCmod.PenMPCC, PenMPCCmod.pen_start!, PenMPCCmod.pen_update!

import OutputALASmod.OutputALAS, OutputALASmod.Update

import MPCCmod.MPCC, MPCCmod.viol_contrainte
import MPCCmod.obj, MPCCmod.grad
import MPCCmod.consG, MPCCmod.consH

import Stopping.TStopping, Stopping.start!, Stopping.stop
import StoppingRelax.TStoppingPAS, StoppingRelax.pas_start!, StoppingRelax.pas_rhoupdate!
import StoppingRelax.pas_stop!, StoppingRelax.ending_test

import RRelaxmod.RRelax, RRelaxmod.relax_start!

import UnconstrainedMPCCActif.LineSearchSolve #ne devrait pas être là

import Relaxation.psi #un peu bizarre ici

import RActifmod.RActif, RActifmod.actif_start!

"""
Type ALASMPCC : 

liste des constructeurs :
+ALASMPCC(mod::MPCCmod.MPCC,
          r::Float64,s::Float64,t::Float64,
          prec::Float64, rho::Vector)

liste des méthodes :


liste des accesseurs :
-addInitialPoint(alas::ALASMPCC,x0::Vector)

liste des fonctions :
+solvePAS(alas::ALASMPCC)

"""
#Remark: (mod,r,s,t,tb) represents the relaxed problem.

type ALASMPCC

 mod      :: MPCC

 #paramètres du pb
 r        :: Float64
 s        :: Float64
 t        :: Float64
 tb       :: Float64

 #paramètres algorithmiques
 prec     :: Float64 #precision à 0
 rho_init :: Vector #nombre >= 0

 xj       :: Vector

 paramset :: ParamSet
 algoset  :: AlgoSet

 spas     :: TStoppingPAS # stopping the penalization-active-set strategy

 rrelax   :: RRelax

end

function ALASMPCC(mod      :: MPCC,
                  r        :: Float64,
                  s        :: Float64,
                  t        :: Float64,
                  prec     :: Float64,
                  rho      :: Vector,
                  paramset :: ParamSet,
                  algoset  :: AlgoSet,
                  x        :: Vector)

 rho_init = rho
 tb = paramset.tb(r,s,t)

 spas = TStoppingPAS(max_iter=paramset.ite_max_alas,
                     atol=prec,goal_viol=paramset.goal_viol,
                     rho_max=paramset.rho_max)

 # x=SlackComplementarityProjection(alas)

 rrelax = RRelax(x)

 return ALASMPCC(mod,r,s,t,tb,prec,rho_init,x,paramset,algoset,spas,rrelax)
end
"""
Accesseur : modifie le point initial
"""

function set_x(alas :: ALASMPCC,
               x0   :: Vector)

 alas.xj = x0

 return alas
end

############################################################################
#
# Update an ALAS
#
############################################################################
function update_alas!(alas :: ALASMPCC,
                      r    :: Float64,
                      s    :: Float64,
                      t    :: Float64,
                      prec :: Float64,
                      rho  :: Vector)

 alas.r = r
 alas.s = s
 alas.t = t
 alas.prec = prec
 alas.rho = rho

 alas.tb = alas.paramset.tb(r,s,t)

 return alas
end

############################################################################
#solvePAS() : méthode de penalisation/activation de contraintes DHKM 16'
#
#Méthode avec gestion à l'intérieur du paramètre de pénalité
############################################################################

include("solve_pas.jl")

##################################################################################
#
# ActifMPCC == PenMPCCSolve
#function solve_subproblem_pen(ma      :: ActifMPCC,
#                              xjk     :: Vector,
#                              oa      :: OutputALAS;
#                              verbose :: Bool = true)
#
##################################################################################

include("solve_subproblem_pen.jl")

############################################################################
#
# Minimisation sans contrainte dans domaine actif
#
############################################################################

include("working_min.jl")

############################################################################
#
# Mise à jour de rho
#
############################################################################

function CheckUpdateRho(alas    :: ALASMPCC,
                        ma      :: ActifMPCC,
                        xjkl    :: Vector,
                        verbose :: Bool)

 ht, gradpen = ma.rpen.fx, ma.rpen.gx
 feas = alas.spas.feasibility
 ρ = ma.pen.ρ
 u = ma.pen.u

 cx = viol_contrainte(alas.mod, xjkl)
 ρ = RhoUpdate(alas, ρ, ma.crho,abs.(cx))

 #on change le problème donc on réinitialise beta
 setcrho(ma,alas.algoset.crho_update(feas,ρ))

 #met à jour la fonction objectif après la mise à jour de rho
 ma.pen.nlp, ht, gradpen = UpdatePenaltyNLP(alas,ρ,xjkl,
                                            u,ma.pen.nlp,objpen=ht,
                                            gradpen=gradpen,crho=ma.crho)

 #met à jour les différents paramètres horizontaux
 setbeta(ma,0.0) #on change le problème donc on réinitialise beta
 #ActifMPCCmod.sethess(ma,H) #on change le problème donc on réinitialise Hess

 ma.rpen.fx, ma.rpen.gx = ht, gradpen

 return ρ,ma
end

############################################################################
#Mise à jour de rho:
#structure de rho :
#2*mod.nb_comp
#length(mod.mp.meta.lvar)
#length(mod.mp.meta.uvar)
#length(mod.mp.meta.lcon)
#length(mod.mp.meta.ucon)
############################################################################

function RhoUpdate(alas :: ALASMPCC,
                   ρ    :: Vector,
                   crho :: Float64,
                   err  :: Vector;
                   l    :: Int64=0)

 nr = length(ρ)

 ρ = min.(ρ*alas.paramset.rho_update, ones(nr)*alas.paramset.rho_max)

 return ρ
end

############################################################################
#
# Create/Update the Penalized NLP after an update in ρ
#
# CreatePenaltyNLP(alas::ALASMPCC,
#                  xj::Vector,ρ::Vector,
#                  usg::Vector,ush::Vector,
#                  uxl::Vector,uxu::Vector,
#                  ucl::Vector,ucu::Vector)
#
# UpdatePenaltyNLP(alas::ALASMPCC,
#                  ρ::Vector,
#                  xj::Vector,
#                  usg::Vector,ush::Vector,
#                  uxl::Vector,uxu::Vector,
#                  ucl::Vector,ucu::Vector,
#                  pen_nlp::NLPModels.AbstractNLPModel;
#                  gradpen::Vector=[],
#                  objpen::Float64=zeros(0),
#                  crho::Float64=1.0)
#
############################################################################

include("nlp_penalty.jl")

############################################################################
# SlackComplementarityProjection : 
# projete le point alas.xj sur la contrainte de complémentarité
# avec les slack
# 
#
# SlackComplementarityProjection(alas::ALASMPCC) --> x :: Vector 
#                                                         (n + 2nb_comp)
#
############################################################################

include("projection_init.jl")

############################################################################
#
# Initialize Lagrange Multipliers of the complementarity constraints
#
############################################################################

function LagrangeCompInit(alas :: ALASMPCC,
                          ρ    :: Vector,
                          xj   :: Vector;
                          c    :: Vector = Float64[])

 n = alas.mod.n
 nb_comp = alas.mod.nb_comp

 ρ_eqg, ρ_eqh, ρ_lvar, ρ_uvar, ρ_lcons, ρ_ucon = RhoDetail(alas.mod, ρ)
 
 if c == Float64[]
  psiyg = psi(xj[n+1:n+nb_comp], alas.r, alas.s, alas.t)
  psiyh = psi(xj[n+1+nb_comp:n+2*nb_comp], alas.r, alas.s, alas.t)
  phi = (xj[n+1:n+nb_comp]-psiyh).*(xj[n+1+nb_comp:n+2*nb_comp]-psiyg)
  slackg = xj[n+1:n+nb_comp]-alas.tb
  slackh = xj[n+1+nb_comp:n+2*nb_comp]-alas.tb
 else
  phi = c[2*nb_comp+1:3*nb_comp]
  slackg = c[1:nb_comp]
  slackh = c[nb_comp+1:2*nb_comp]
 end

 lphi = max.(phi,zeros(nb_comp))
 lg = max.(-ρ_eqg.*(slackg), zeros(nb_comp))
 lh = max.(-ρ_eqh.*(slackh), zeros(nb_comp))

 return [lg;lh;lphi]
end


############################################################################
#
# Initialize Lagrange Multipliers of the classical constraints
#
############################################################################

function LagrangeInit(alas :: ALASMPCC,
                      ρ    :: Vector,
                      xj   :: Vector;
                      c    :: Vector = Float64[])

 n = alas.mod.n
 ncon = alas.mod.mp.meta.ncon
 nb_comp = alas.mod.nb_comp

 ρ_eqg, ρ_eqh, ρ_lvar, ρ_uvar, ρ_lcon, ρ_ucon = RhoDetail(alas.mod, ρ)

 if c == Float64[]

  uxl = max.(ρ_lvar.*(alas.mod.mp.meta.lvar-xj[1:n]), zeros(n))
  uxu = max.(ρ_uvar.*(xj[1:n]-alas.mod.mp.meta.uvar), zeros(n))

  if alas.mod.mp.meta.ncon != 0

   cx = NLPModels.cons(alas.mod.mp, xj[1:n])
   nc = length(alas.mod.mp.meta.y0) #nombre de contraintes

   ucl = max.(ρ_lcon.*(alas.mod.mp.meta.lcon-cx), zeros(nc))
   ucu = max.(ρ_ucon.*(cx-alas.mod.mp.meta.ucon), zeros(nc))

  else

   ucl = []
   ucu = []

  end

  usg = zeros(alas.mod.nb_comp)
  ush = zeros(alas.mod.nb_comp)

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

function LagrangeUpdate(alas :: ALASMPCC,
                        ρ    :: Vector,
                        xjk  :: Vector,
                        u    :: Vector)

 n = alas.mod.n

 ρ_eqg, ρ_eqh, ρ_lvar, ρ_uvar, ρ_lcon, ρ_ucon = RhoDetail(alas.mod, ρ)
 usg,ush,uxl,uxu,ucl,ucu = RhoDetail(alas.mod,u)

 uxl = uxl+max.(ρ_lvar.*(xjk[1:n]-alas.mod.mp.meta.uvar), -uxl)
 uxu = uxu+max.(ρ_uvar.*(alas.mod.mp.meta.lvar-xjk[1:n]), -uxu)

 if alas.mod.mp.meta.ncon != 0

  c = NLPModels.cons(alas.mod.mp, xjk[1:n])
  ucl = ucl+max.(ρ_lcon.*(c-alas.mod.mp.meta.ucon), -ucl)
  ucu = ucu+max.(ρ_ucon.*(alas.mod.mp.meta.lcon-c), -ucu)

 end

 if alas.mod.nb_comp != 0

  G(x) = consG(alas.mod, x)
  H(x) = consH(alas.mod, x)
  usg = usg+ρ_eqg.*(G(xjk[1:n])-xjk[n+1:n+alas.mod.nb_comp])
  ush = ush+ρ_eqh.*(H(xjk[1:n])-xjk[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 end

 return vcat(usg,ush,uxl,uxu,ucl,ucu)
end

############################################################################
#
# Initialize ActifMPCC
#
# InitializePenMPCC(alas::ALASMPCC,
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
# InitializeMPCCActif(alas::ALASMPCC,
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

include("init_actifmpcc.jl")

############################################################################
#
# Give the detail of the vector Rho:
# return ρ_eq, ρ_ineq_var, ρ_ineq_cons
#
############################################################################

function RhoDetail(mod :: MPCC,
                   ρ   :: Vector)

 nil = length(mod.mp.meta.lvar)
 niu = length(mod.mp.meta.uvar)
 nlcon = length(mod.mp.meta.lcon)
 nucon = length(mod.mp.meta.ucon)
 nb_comp = mod.nb_comp

 return ρ[1:nb_comp],
        ρ[nb_comp+1:2*nb_comp],
        ρ[2*nb_comp+1:2*nb_comp+nil],
        ρ[2*nb_comp+nil+1:2*nb_comp+nil+niu],
        ρ[2*nb_comp+nil+niu+1:2*nb_comp+nil+niu+nlcon],
        ρ[2*nb_comp+nil+niu+nlcon+1:2*nb_comp+nil+niu+nlcon+nucon]
end

#end of module
end
