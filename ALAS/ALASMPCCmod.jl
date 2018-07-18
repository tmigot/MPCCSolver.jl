module ALASMPCCmod

using NLPModels

import AlgoSetmod.AlgoSet
import ParamSetmod.ParamSet

import ActifMPCCmod.ActifMPCC,ActifMPCCmod.grad
import ActifMPCCmod.ActiveCons
import ActifMPCCmod.LSQComputationMultiplierBool,ActifMPCCmod.RelaxationRule
import ActifMPCCmod.setbeta, ActifMPCCmod.setcrho #pas sûr que ça reste

import OutputALASmod.OutputALAS, OutputALASmod.Update

import MPCCmod.MPCC, MPCCmod.viol_contrainte

import Stopping.TStopping, Stopping.start!, Stopping.stop
import Stopping.TStoppingPAS, Stopping.pas_start!, Stopping.pas_rhoupdate!
import Stopping.pas_stop!, Stopping.ending_test

import UnconstrainedMPCCActif.LineSearchSolve #ne devrait pas être là

import Relaxation.psi #un peu bizarre ici

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

-SlackComplementarityProjection(alas::ALASMPCC)
-LagrangeInit(alas::ALASMPCC,rho::Vector,xj::Vector)
-LagrangeCompInit(alas::ALASMPCC,rho::Vector,xj::Vector)
-LagrangeUpdate(alas::ALASMPCC,rho::Vector,xjk::Vector,
                        uxl::Vector,uxu::Vector,
                        ucl::Vector,ucu::Vector,
                        usg::Vector,ush::Vector)
Penaltygen(alas::ALASMPCCmod.ALASMPCC,
                    x::Vector,yg::Vector,yh::Vector,
                    rho::Vector,usg::Vector,ush::Vector,
                    uxl::Vector,uxu::Vector,
                    ucl::Vector,ucu::Vector)
GradPenaltygen(alas::ALASMPCCmod.ALASMPCC,
                        x::Vector,yg::Vector,yh::Vector,
                        rho::Vector,usg::Vector,ush::Vector,
                        uxl::Vector,uxu::Vector,
                        ucl::Vector,ucu::Vector)
HessPenaltygen(alas::ALASMPCCmod.ALASMPCC,
                        x::Vector,yg::Vector,yh::Vector,
                        rho::Vector,usg::Vector,ush::Vector,
                        uxl::Vector,uxu::Vector,
                        ucl::Vector,ucu::Vector)
ObjGradPenaltygen(alas::ALASMPCCmod.ALASMPCC,
                            x::Vector,yg::Vector,yh::Vector,
                            rho::Vector,usg::Vector,ush::Vector,
                            uxl::Vector,uxu::Vector,
                            ucl::Vector,ucu::Vector)
ObjGradHessPenaltygen(alas::ALASMPCCmod.ALASMPCC,
                            x::Vector,yg::Vector,yh::Vector,
                            rho::Vector,usg::Vector,ush::Vector,
                            uxl::Vector,uxu::Vector,
                            ucl::Vector,ucu::Vector)
InitializeMPCCActif(alas::ALASMPCCmod.ALASMPCC,
                             xj::Vector,
                             rho::Vector,
                             usg::Vector,
                             ush::Vector,
                             uxl::Vector,
                             uxu::Vector,
                             ucl::Vector,
                             ucu::Vector)
function CreatePenaltyNLP(alas::ALASMPCCmod.ALASMPCC,
                          xj::Vector,rho::Vector,
                          usg::Vector,ush::Vector,
                          uxl::Vector,uxu::Vector,
                          ucl::Vector,ucu::Vector)
function UpdatePenaltyNLP(alas::ALASMPCCmod.ALASMPCC,
                          rho::Vector,xj::Vector,
                          usg::Vector,ush::Vector,
                          uxl::Vector,uxu::Vector,
                          ucl::Vector,ucu::Vector,
                          pen_nlp::NLPModels.AbstractNLPModel;
                          gradpen::Vector=[],objpen::Float64=zeros(0),crho::Float64=1.0)
RhoUpdate(alas::ALASMPCC,rho::Vector,crho::Float64,err::Vector;l::Int64=0)
RhoDetail(alas::ALASMPCC,rho::Vector)
"""

type ALASMPCC

 mod::MPCC

 #paramètres du pb
 r::Float64
 s::Float64
 t::Float64
 tb::Float64

 #paramètres algorithmiques
 prec::Float64 #precision à 0
 rho_init::Vector #nombre >= 0

 xj::Vector

 paramset::ParamSet
 algoset::AlgoSet
 #function de minimisation

 sts::TStopping
 spas::TStoppingPAS

end

function ALASMPCC(mod::MPCC,
                  r::Float64,
                  s::Float64,
                  t::Float64,
                  prec::Float64,
                  rho::Vector,
                  paramset::ParamSet,
                  algoset::AlgoSet,
                  x::Vector)

 rho_init=rho
 tb=paramset.tb(r,s,t)

 sts=TStopping(max_iter=paramset.ite_max_viol,atol=prec)
 spas=TStoppingPAS(max_iter=paramset.ite_max_alas,
                   atol=prec,goal_viol=paramset.goal_viol,
                   rho_max=paramset.rho_max)

 return ALASMPCC(mod,r,s,t,tb,prec,rho_init,x,paramset,algoset,sts,spas)
end
"""
Accesseur : modifie le point initial
"""

function addInitialPoint(alas::ALASMPCC,x0::Vector)

 alas.xj=x0

 return alas
end

#######################
#solvePAS() : méthode de penalisation/activation de contraintes DHKM 16'
#
#Méthode avec gestion à l'intérieur du paramètre de pénalité
######################

include("solve_pas.jl")

######################
#
# Minimisation sans contrainte
#
######################

include("working_min.jl")

######################
#
# Mise à jour de rho
#
######################

function CheckUpdateRho(alas::ALASMPCC,ma::ActifMPCC,
                        xjkl::Vector,ρ::Vector,feas::Float64,
                        usg::Vector,ush::Vector,
                        uxl::Vector,uxu::Vector,
                        ucl::Vector,ucu::Vector,
                        ht::Float64,gradpen::Vector,
                        verbose::Bool)

  #Conditionnelle: Mise à jour de \rho
  #if (l==l_max || small_step || !Armijosuccess || dual_feasible) && (feas>obj_viol*feask && !feasible)
  #if feas>obj_viol*feask && !feasible

   ρ=RhoUpdate(alas,ρ,ma.crho,abs.(viol_contrainte(alas.mod,xjkl)))
   #on change le problème donc on réinitialise beta
   setcrho(ma,alas.algoset.crho_update(feas,ρ))

   #met à jour la fonction objectif après la mise à jour de rho
   ma.nlp,ht,gradpen=UpdatePenaltyNLP(alas,ρ,xjkl,
                                      usg,ush,uxl,uxu,ucl,ucu,
                                      ma.nlp,objpen=ht,
                                      gradpen=gradpen,crho=ma.crho)
   setbeta(ma,0.0) #on change le problème donc on réinitialise beta
   #ActifMPCCmod.sethess(ma,H) #on change le problème donc on réinitialise Hess

  #end
 return ρ,ma,ht,gradpen
end

"""
Mise à jour de rho:
structure de rho :
2*mod.nb_comp
length(mod.mp.meta.lvar)
length(mod.mp.meta.uvar)
length(mod.mp.meta.lcon)
length(mod.mp.meta.ucon)
"""
function RhoUpdate(alas::ALASMPCC,ρ::Vector,crho::Float64,err::Vector;l::Int64=0)

 nr = length(ρ)
 #rho[find(x->x>0,err)]*=alas.rho_update
 #ρ*=alas.mod.paramset.rho_update
 #ρ=min.(ρ*alas.paramset.rho_update,ones(nr)*alas.paramset.rho_max*crho)
 ρ=min.(ρ*alas.paramset.rho_update,ones(nr)*alas.paramset.rho_max)

 return ρ
end

#######################
#SlackComplementarityProjection : 
#projete le point x0 sur la contrainte de complémentarité avec les slack
#pré-requis : x0 doit être de taille n+2q
#######################

include("projection_init.jl")

"""
Initialize Lagrange Multipliers of the active constraints
"""

function LagrangeCompInit(alas::ALASMPCC,ρ::Vector,xj::Vector)

 n=alas.mod.n
 nb_comp=alas.mod.nb_comp
 ρ_eqg,ρ_eqh,ρ_ineq_lvar,ρ_ineq_uvar,ρ_ineq_lcons,ρ_ineq_ucons=RhoDetail(alas.mod,ρ)
 
 psiyg=psi(xj[n+1:n+nb_comp],alas.r,alas.s,alas.t)
 psiyh=psi(xj[n+1+nb_comp:n+2*nb_comp],alas.r,alas.s,alas.t)
 phi=(xj[n+1:n+nb_comp]-psiyh).*(xj[n+1+nb_comp:n+2*nb_comp]-psiyg)

 lphi=max.(phi,zeros(nb_comp))
 lg=max.(-ρ_eqg.*(xj[n+1:n+nb_comp]-alas.tb),zeros(nb_comp))
 lh=max.(-ρ_eqh.*(xj[n+1+nb_comp:n+2*nb_comp]-alas.tb),zeros(nb_comp))

 return [lg;lh;lphi]
end


"""
Initialisation des multiplicateurs de Lagrange
"""
function LagrangeInit(alas::ALASMPCC,ρ::Vector,xj::Vector)

 n=alas.mod.n
 ρ_eqg,ρ_eqh,ρ_ineq_lvar,ρ_ineq_uvar,ρ_ineq_lcons,ρ_ineq_ucons=RhoDetail(alas.mod,ρ)

 uxl=max.(ρ_ineq_lvar.*(alas.mod.mp.meta.lvar-xj[1:n]),zeros(n))
 uxu=max.(ρ_ineq_uvar.*(xj[1:n]-alas.mod.mp.meta.uvar),zeros(n))

 if alas.mod.mp.meta.ncon!=0

  c(z)=NLPModels.cons(alas.mod.mp,z)
  nc=length(alas.mod.mp.meta.y0) #nombre de contraintes

  ucl=max.(ρ_ineq_lcons.*(alas.mod.mp.meta.lcon-c(xj[1:n])),zeros(nc))
  ucu=max.(ρ_ineq_ucons.*(c(xj[1:n])-alas.mod.mp.meta.ucon),zeros(nc))

 else

  ucl=[];ucu=[];

 end

 usg=zeros(alas.mod.nb_comp)
 ush=zeros(alas.mod.nb_comp)

 return uxl,uxu,ucl,ucu,usg,ush
end

"""
Mise à jour des multiplicateurs de Lagrange
"""
function LagrangeUpdate(alas::ALASMPCC,ρ::Vector,xjk::Vector,
                        uxl::Vector,uxu::Vector,
                        ucl::Vector,ucu::Vector,
                        usg::Vector,ush::Vector)

 n=alas.mod.n
 ρ_eqg,ρ_eqh,ρ_ineq_lvar,ρ_ineq_uvar,ρ_ineq_lcons,ρ_ineq_ucons=RhoDetail(alas.mod,ρ)

 uxl=uxl+max.(ρ_ineq_lvar.*(xjk[1:n]-alas.mod.mp.meta.uvar),-uxl)
 uxu=uxu+max.(ρ_ineq_uvar.*(alas.mod.mp.meta.lvar-xjk[1:n]),-uxu)
 if alas.mod.mp.meta.ncon!=0
  c=NLPModels.cons(alas.mod.mp,xjk[1:n])
  ucl=ucl+max.(ρ_ineq_lcons.*(c-alas.mod.mp.meta.ucon),-ucl)
  ucu=ucu+max.(ρ_ineq_ucons.*(alas.mod.mp.meta.lcon-c),-ucu)
 end

 if alas.mod.nb_comp!=0
  G(x)=NLPModels.cons(alas.mod.G,x)
  H(x)=NLPModels.cons(alas.mod.H,x)
  usg=usg+ρ_eqg.*(G(xjk[1:n])-xjk[n+1:n+alas.mod.nb_comp])
  ush=ush+ρ_eqh.*(H(xjk[1:n])-xjk[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])
 end

 return uxl,uxu,ucl,ucu,usg,ush
end

####
#
# Initialize ActifMPCC
#
####

include("init_actifmpcc.jl")

"""
Renvoie : rho_eq, rho_ineq_var, rho_ineq_cons
"""
function RhoDetail(mod::MPCC,ρ::Vector)

 nb_ineq_lvar=length(mod.mp.meta.lvar)
 nb_ineq_uvar=length(mod.mp.meta.uvar)
 nb_ineq_lcons=length(mod.mp.meta.lcon)
 nb_ineq_ucons=length(mod.mp.meta.ucon)
 nb_comp=mod.nb_comp

 return ρ[1:nb_comp],
        ρ[nb_comp+1:2*nb_comp],
        ρ[2*nb_comp+1:2*nb_comp+nb_ineq_lvar],
        ρ[2*nb_comp+nb_ineq_lvar+1:2*nb_comp+nb_ineq_lvar+nb_ineq_uvar],
        ρ[2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+1:2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+nb_ineq_lcons],
        ρ[2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+nb_ineq_lcons+1:2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+nb_ineq_lcons+nb_ineq_ucons]
end

#end of module
end
