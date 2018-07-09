module ALASMPCCmod

using OutputALASmod
using ActifMPCCmod
using UnconstrainedMPCCActif
using MPCCmod
using Relaxation

using NLPModels

include("RelaxedNLPenalization.jl")

"""
Type ALASMPCC : 

liste des constructeurs :
ALASMPCC(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64, prec::Float64)
ALASMPCC(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64, prec::Float64, rho::Vector)

liste des méthodes :


liste des accesseurs :


liste des fonctions :
solvePAS(alas::ALASMPCC)

EndingTest(alas::ALASMPCC,Armijosuccess::Bool,small_step::Bool,feas::Float64,dual_feas::Float64,k::Int64)
SlackComplementarityProjection(alas::ALASMPCC)
LagrangeInit(alas::ALASMPCC,rho::Vector,xj::Vector)
LagrangeCompInit(alas::ALASMPCC,rho::Vector,xj::Vector)
LagrangeUpdate(alas::ALASMPCC,rho::Vector,xjk::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector,usg::Vector,ush::Vector)
Penaltygen(alas::ALASMPCCmod.ALASMPCC,x::Vector,yg::Vector,yh::Vector,rho::Vector,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector)
RhoUpdate(rho::Vector,err::Vector,alas::ALASMPCC)
RhoDetail(rho::Vector,alas::ALASMPCC)
"""

type ALASMPCC
 mod::MPCCmod.MPCC
 #paramètres du pb
 r::Float64
 s::Float64
 t::Float64
 tb::Float64

 #paramètres algorithmiques
 prec::Float64 #precision à 0
 rho_init::Vector #nombre >= 0
end

function ALASMPCC(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64, prec::Float64, rho::Vector)

 rho_init=rho
 tb=mod.paramset.tb(r,s,t)

 return ALASMPCC(mod,r,s,t,tb,prec,rho_init)
end

"""
solvePAS() : méthode de penalisation/activation de contraintes DHKM 16'

Méthode avec gestion à l'intérieur du paramètre de pénalité
"""
function solvePAS(alas::ALASMPCC; verbose::Bool=true)

#Initialisation paramètres :
 rho=alas.rho_init
 k_max=alas.mod.paramset.ite_max_alas
 l_max=alas.mod.paramset.ite_max_viol
 rho_update=alas.mod.paramset.rho_update
 obj_viol=alas.mod.paramset.goal_viol

 n=length(alas.mod.mp.meta.x0)

 gradpen=Vector(n)
 gradpen_prec=Vector(n)
 xj=Vector(n+2*alas.mod.nb_comp)
 xjk=Vector(n+2*alas.mod.nb_comp)
 xjkl=Vector(n+2*alas.mod.nb_comp)

# S0 : initialisation du problème avec slack (projection sur _|_ )
 xj=SlackComplementarityProjection(alas)

 #variables globales en sortie du LineSearch
 step=1.0
 Armijosuccess=true
 wnew=zeros(Bool,0,0)

 #Initialisation multiplicateurs de la contrainte de complémentarité
 lg,lh,lphi=LagrangeCompInit(alas,rho,xj)

 # Initialisation multiplicateurs : uxl,uxu,ucl,ucu :
 uxl,uxu,ucl,ucu,usg,ush=LagrangeInit(alas,rho,xj)

# S1 : Initialisation du MPCC_Actif 
 ma=InitializeMPCCActif(alas,xj,rho,usg,ush,uxl,uxu,ucl,ucu)

# S2 : Major Loop Activation de contrainte
 #initialisation
 k=0
 xjk=xj
 small_step=false
 ht,gradpen=NLPModels.objgrad(ma.nlp,xjk)

 #direction initial
 dj=zeros(n+2*alas.mod.nb_comp)

 l_negative=findfirst(x->x<0,[lg;lh;lphi])!=0

 feas=MPCCmod.viol_contrainte_norm(alas.mod,xjk)
 feasible=feas<=alas.prec

 dual_feas=norm(ActifMPCCmod.grad(ma,xjk,gradpen),Inf)
 multi_norm=alas.mod.algoset.scaling_dual(usg,ush,uxl,uxu,ucl,ucu,
                                          lg,lh,lphi,alas.prec,
                                          alas.mod.paramset.precmpcc,
                                          norm(rho,Inf),dual_feas)
 dual_feasible=dual_feas/multi_norm<=alas.prec

 oa=OutputALASmod.OutputALAS(xjk,dj,feas,dual_feas,rho,ht)

 while k<k_max && (l_negative || !((feasible || minimum(rho)==alas.mod.paramset.rho_max*ma.crho ) && dual_feasible)) && Armijosuccess && !small_step

  l=0
  xjkl=xjk
  feask=feas
  gradpen_prec=gradpen

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#BLOC EN TRAVAUX
# Début du RelaxedNL.RelaxedNLsolve(alas,xjk,feas,gradpen)
# ma,xjk,ht,gradpen,uxl,uxu,ucl,ucu,usg,ush,rho,Armijosuccess,small_step,feasible,dual_feasible=RelaxedNLsolvePenalization(alas,ma,xjk,feas,feasible,dual_feas,dual_feasible,gradpen,ht,dj,step,oa)

######

#boucle pour augmenter le paramètre de pénalisation tant qu'on a pas baissé suffisament la violation
  while l<l_max && !dual_feasible && Armijosuccess && !small_step
   #Unconstrained Solver modifié pour résoudre le sous-problème avec contraintes de bornes
   xjkl,ma.w,dj,step,wnew,outputArmijo,small_step,ols,gradpen,ht=UnconstrainedMPCCActif.LineSearchSolve(ma,xjkl,dj,step,gradpen,ht)
   feas=MPCCmod.viol_contrainte_norm(alas.mod,xjkl);
   feasible=feas<=alas.prec

   dual_feas=norm(ActifMPCCmod.grad(ma,xjkl,gradpen),Inf)
   dual_feasible=dual_feas<=alas.mod.algoset.unconstrained_stopping(alas.prec,rho)

   OutputALASmod.Update(oa,xjkl,rho,feas,dual_feas,dj,step,ols,ht,n)

   Armijosuccess=(outputArmijo==0)
   if outputArmijo==2
    @show "Unbounded subproblem"
    return xj,EndingTest(alas,Armijosuccess,small_step,feas,dual_feas,k),rho,oa
   end
   l+=1
  end

  #On met à jour rho si ça n'a pas été:
  if (l==l_max || small_step || !Armijosuccess || dual_feasible) && (feas>obj_viol*feask && !feasible)
   verbose && print_with_color(:red, "Max ité Unc. Min. l=$l |x|=$(norm(xjkl,Inf)) |c(x)|=$feas |L'|=$dual_feas Arm=$Armijosuccess small_step=$small_step rho=$(norm(rho,Inf))  \n")
   rho=RhoUpdate(alas,rho,ma.crho,abs(MPCCmod.viol_contrainte(alas.mod,xjkl)))
   ActifMPCCmod.setcrho(ma,alas.mod.algoset.crho_update(feas,rho)) #on change le problème donc on réinitialise beta

   #met à jour la fonction objectif après la mise à jour de rho
   ma.nlp,ht,gradpen=UpdatePenaltyNLP(alas,rho,xjkl,usg,ush,uxl,uxu,ucl,ucu,ma.nlp,objpen=ht,gradpen=gradpen,crho=ma.crho)
   ActifMPCCmod.setbeta(ma,0.0) #on change le problème donc on réinitialise beta
   #ActifMPCCmod.sethess(ma,H) #on change le problème donc on réinitialise Hess
   dual_feas=norm(ActifMPCCmod.grad(ma,xjkl,gradpen),Inf)
   dual_feasible=dual_feas<=alas.prec
  end
  xjk=xjkl

  #Mise à jour des paramètres de la pénalité Lagrangienne (uxl,uxu,ucl,ucu)
  uxl,uxu,ucl,ucu,usg,ush=LagrangeUpdate(alas,rho,xjk,uxl,uxu,ucl,ucu,usg,ush)

  #met à jour la fonction objectif après la mise à jour des multiplicateurs
  temp,ht,gradpen=UpdatePenaltyNLP(alas,rho,xjk,usg,ush,uxl,uxu,ucl,ucu,
                                   ma.nlp,objpen=ht,gradpen=gradpen)

  multi_norm=alas.mod.algoset.scaling_dual(usg,ush,uxl,uxu,ucl,ucu,lg,lh,lphi,alas.prec,alas.mod.paramset.precmpcc,norm(rho,Inf),dual_feas)

  feasible=feas<=alas.prec
  dual_feasible=dual_feas/multi_norm<=alas.prec


#FIN du BLOC RESOLUTION NON-LINEAIRE
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  #Mise à jour des multiplicateurs de la complémentarité
  if alas.mod.nb_comp>0
   lg,lh,lphi=ActifMPCCmod.LSQComputationMultiplier(ma,gradpen,xjk)
   l_negative=findfirst(x->x<0,[lg;lh;lphi])!=0 #vrai si un multiplicateur est negatif
  end

  #si on bloque mais qu'un multiplicateur est <0 on continue
  if l_negative && (!Armijosuccess || small_step) 
   Armijosuccess=true
   small_step=false
  end

  #Relaxation rule si on a fait un pas de Armijo-Wolfe et si un multiplicateur est du mauvais signe
  if k!=0 && (dot(gradpen,dj)>=alas.mod.paramset.tau_wolfe*dot(gradpen_prec,dj) || step==0.0) && findfirst(x->x<0,[lg;lh;lphi])!=0

   ActifMPCCmod.RelaxationRule(ma,xjk,lg,lh,lphi,wnew)
   if findfirst(x->x<0,[lg;lh;lphi])!=0
    dual_feasible=false #si on a relaché des contraintes, on force à faire une itération
   end
   verbose && print_with_color(:yellow, "Active set: $(ma.wcc) \n")
  end

  k+=1
  verbose && k>=k_max && print_with_color(:red, "Max ité. Lagrangien \n")
 end

#Traitement finale :
 xj=xjk

 stat=EndingTest(alas,Armijosuccess,small_step,feas,dual_feas,k)

 return xj,stat,rho,oa
end

function EndingTest(alas::ALASMPCC,Armijosuccess::Bool,small_step::Bool,feas::Float64,dual_feas::Float64,k::Int64)

 stat=0
 if !Armijosuccess
  print_with_color(:red, "Failure : Armijo Failure\n")
  stat=1
 end
 if small_step
  print_with_color(:red, "Failure : too small step\n")
  #stat=1
 end
 if feas>alas.prec
  print_with_color(:red, "Failure : Infeasible Solution. norm: $feas\n")
  #stat=1
 end
 if dual_feas>alas.prec
  if k>=alas.mod.paramset.ite_max_alas
   print_with_color(:red, "Failure : Non-optimal Sol. norm: $dual_feas\n")
   stat=1
  else
   print_with_color(:red, "Inexact : Fritz-John Sol. norm: $dual_feas\n")
   #stat=0
  end
 end

 return stat
end

function Feasible(alas,x::Vector)
 return MPCCmod.viol_contrainte_norm(alas.mod,x)<=alas.prec
end
function DualFeasible(alas,gradpen::Vector)
 return norm(gradpen[1:n],Inf)<=alas.prec
end

"""
SlackComplementarityProjection : 
projete le point x0 sur la contrainte de complémentarité avec les slack
pré-requis : x0 doit être de taille n+2q
"""
function SlackComplementarityProjection(alas::ALASMPCC)

 nb_comp=alas.mod.nb_comp

 if nb_comp==0
  return alas.mod.xj
 end

 #initialisation :
 #x=[alas.mod.xj;alas.mod.G(alas.mod.xj);alas.mod.H(alas.mod.xj)]
 x=[alas.mod.xj;NLPModels.cons(alas.mod.G,alas.mod.xj);NLPModels.cons(alas.mod.H,alas.mod.xj)]
 n=length(x)-2*nb_comp

 #projection sur les contraintes de positivité relaxés : yG>=tb et yH>=tb
 x[n+1:n+nb_comp]=max.(x[n+1:n+nb_comp],ones(nb_comp)*alas.tb)
 x[n+nb_comp+1:n+2*nb_comp]=max.(x[n+nb_comp+1:n+2*nb_comp],ones(nb_comp)*alas.tb)

 #projection sur les contraintes papillons : yG<=psi(yH,r,s,t) et yH<=psi(yG,r,s,t)
 for i=1:nb_comp
  psiyg=Relaxation.psi(x[n+i],alas.r,alas.s,alas.t)
  psiyh=Relaxation.psi(x[n+nb_comp+i],alas.r,alas.s,alas.t)

  if x[n+i]-psiyh>0 && x[n+nb_comp+i]-psiyg>0
   x[n+i]>=x[n+nb_comp+i] ? x[n+nb_comp+i]=psiyg : x[n+i]=psiyh
  end
 end

 return x
end

"""
Initialisation des multiplicateurs de Lagrange
"""
function LagrangeInit(alas::ALASMPCC,rho::Vector,xj::Vector)

  n=length(alas.mod.mp.meta.x0)
  rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=RhoDetail(alas,rho)

  uxl=max.(rho_ineq_lvar.*(alas.mod.mp.meta.lvar-xj[1:n]),zeros(n))
  uxu=max.(rho_ineq_uvar.*(xj[1:n]-alas.mod.mp.meta.uvar),zeros(n))
  if alas.mod.mp.meta.ncon!=0
   c(z)=NLPModels.cons(alas.mod.mp,z)
   nc=length(alas.mod.mp.meta.y0) #nombre de contraintes

   ucl=max.(rho_ineq_lcons.*(alas.mod.mp.meta.lcon-c(xj[1:n])),zeros(nc))
   ucu=max.(rho_ineq_ucons.*(c(xj[1:n])-alas.mod.mp.meta.ucon),zeros(nc))
  else
   ucl=[];ucu=[];
  end

  usg=zeros(alas.mod.nb_comp)
  ush=zeros(alas.mod.nb_comp)

 return uxl,uxu,ucl,ucu,usg,ush
end

function LagrangeCompInit(alas::ALASMPCC,rho::Vector,xj::Vector)

 n=length(alas.mod.mp.meta.x0)
 nb_comp=alas.mod.nb_comp
 rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=RhoDetail(alas,rho)
 
 psiyg=Relaxation.psi(xj[n+1:n+nb_comp],alas.r,alas.s,alas.t)
 psiyh=Relaxation.psi(xj[n+1+nb_comp:n+2*nb_comp],alas.r,alas.s,alas.t)
 phi=(xj[n+1:n+nb_comp]-psiyh).*(xj[n+1+nb_comp:n+2*nb_comp]-psiyg)

 lphi=max.(phi,zeros(nb_comp))
 lg=max.(-rho_eqg.*(xj[n+1:n+nb_comp]-alas.tb),zeros(nb_comp))
 lh=max.(-rho_eqh.*(xj[n+1+nb_comp:n+2*nb_comp]-alas.tb),zeros(nb_comp))

 return lg,lh,lphi
end

"""
Mise à jour des multiplicateurs de Lagrange
"""
function LagrangeUpdate(alas::ALASMPCC,rho::Vector,xjk::Vector,
                        uxl::Vector,uxu::Vector,
                        ucl::Vector,ucu::Vector,
                        usg::Vector,ush::Vector)

  n=length(alas.mod.mp.meta.x0)
   rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=RhoDetail(alas,rho)

  uxl=uxl+max.(rho_ineq_lvar.*(xjk[1:n]-alas.mod.mp.meta.uvar),-uxl)
  uxu=uxu+max.(rho_ineq_uvar.*(alas.mod.mp.meta.lvar-xjk[1:n]),-uxu)
  if alas.mod.mp.meta.ncon!=0
   c=NLPModels.cons(alas.mod.mp,xjk[1:n])
   ucl=ucl+max(rho_ineq_lcons.*(c-alas.mod.mp.meta.ucon),-ucl)
   ucu=ucu+max(rho_ineq_ucons.*(alas.mod.mp.meta.lcon-c),-ucu)
  end

  if alas.mod.nb_comp!=0
   G(x)=NLPModels.cons(alas.mod.G,x)
   H(x)=NLPModels.cons(alas.mod.H,x)
   usg=usg+rho_eqg.*(G(xjk[1:n])-xjk[n+1:n+alas.mod.nb_comp])
   ush=ush+rho_eqh.*(H(xjk[1:n])-xjk[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])
  end

 return uxl,uxu,ucl,ucu,usg,ush
end


"""
Fonction de pénalité générique :
"""
function Penaltygen(alas::ALASMPCCmod.ALASMPCC,
                    x::Vector,yg::Vector,yh::Vector,
                    rho::Vector,usg::Vector,ush::Vector,
                    uxl::Vector,uxu::Vector,
                    ucl::Vector,ucu::Vector)
 nb_comp=alas.mod.nb_comp

 err=MPCCmod.viol_contrainte(alas.mod,x,yg,yh)

 return alas.mod.algoset.penalty(alas.mod.mp,err,alas.mod.nb_comp,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
end

function GradPenaltygen(alas::ALASMPCCmod.ALASMPCC,
                        x::Vector,yg::Vector,yh::Vector,
                        rho::Vector,usg::Vector,ush::Vector,
                        uxl::Vector,uxu::Vector,
                        ucl::Vector,ucu::Vector)
 g=Vector(x)
 err=MPCCmod.viol_contrainte(alas.mod,x,yg,yh)

 return alas.mod.algoset.penalty(alas.mod.mp,alas.mod.G,alas.mod.H,
                                 err,alas.mod.nb_comp,
                                 x,yg,yh,rho,
                                 usg,ush,uxl,uxu,ucl,ucu,g)
end

function HessPenaltygen(alas::ALASMPCCmod.ALASMPCC,
                        x::Vector,yg::Vector,yh::Vector,
                        rho::Vector,usg::Vector,ush::Vector,
                        uxl::Vector,uxu::Vector,
                        ucl::Vector,ucu::Vector)
 n=length(alas.mod.mp.meta.x0)
 nb_comp=alas.mod.nb_comp

 Hess=zeros(0,0)
 Hf=tril([NLPModels.hess(alas.mod.mp,x) zeros(n,2*nb_comp);zeros(2*nb_comp,n) eye(2*nb_comp)])

 err=MPCCmod.viol_contrainte(alas.mod,x,yg,yh)

 return Hf+alas.mod.algoset.penalty(alas.mod.mp,alas.mod.G,alas.mod.H,
                                 err,nb_comp,
                                 x,yg,yh,rho,
                                 usg,ush,uxl,uxu,ucl,ucu,Hess)
end

function ObjGradPenaltygen(alas::ALASMPCCmod.ALASMPCC,
                            x::Vector,yg::Vector,yh::Vector,
                            rho::Vector,usg::Vector,ush::Vector,
                            uxl::Vector,uxu::Vector,
                            ucl::Vector,ucu::Vector)
 n=length(x)
 nb_comp=alas.mod.nb_comp

 err=MPCCmod.viol_contrainte(alas.mod,x,yg,yh)

 f=NLPModels.obj(alas.mod.mp,x)
 f+=alas.mod.algoset.penalty(alas.mod.mp,err,alas.mod.nb_comp,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 g=vec([NLPModels.grad(alas.mod.mp,x)' zeros(2*alas.mod.nb_comp)'])
 g+=alas.mod.algoset.penalty(alas.mod.mp,alas.mod.G,alas.mod.H,
                                 err,alas.mod.nb_comp,
                                 x,yg,yh,rho,
                                 usg,ush,uxl,uxu,ucl,ucu,Array{Float64}(0))

 return f,g
end

function ObjGradHessPenaltygen(alas::ALASMPCCmod.ALASMPCC,
                            x::Vector,yg::Vector,yh::Vector,
                            rho::Vector,usg::Vector,ush::Vector,
                            uxl::Vector,uxu::Vector,
                            ucl::Vector,ucu::Vector)
 n=length(x)
 nb_comp=alas.mod.nb_comp

 err=MPCCmod.viol_contrainte(alas.mod,x,yg,yh)

 f=NLPModels.obj(alas.mod.mp,x)
 f+=alas.mod.algoset.penalty(alas.mod.mp,err,alas.mod.nb_comp,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 g=vec([NLPModels.grad(alas.mod.mp,x)' zeros(2*alas.mod.nb_comp)'])
 g+=alas.mod.algoset.penalty(alas.mod.mp,alas.mod.G,alas.mod.H,
                                 err,alas.mod.nb_comp,
                                 x,yg,yh,rho,
                                 usg,ush,uxl,uxu,ucl,ucu,g)
 Hess=tril([NLPModels.hess(alas.mod.mp,x) zeros(n,n);zeros(2*nb_comp,n) eye(2*nb_comp)])
 Hess+=alas.mod.algoset.penalty(alas.mod.mp,alas.mod.G,alas.mod.H,
                                 err,nb_comp,
                                 x,yg,yh,rho,
                                 usg,ush,uxl,uxu,ucl,ucu,Array{Float64}(0,0))

 return f,g,Hess
end

function InitializeMPCCActif(alas::ALASMPCCmod.ALASMPCC,
                             xj::Vector,
                             rho::Vector,
                             usg::Vector,
                             ush::Vector,
                             uxl::Vector,
                             uxu::Vector,
                             ucl::Vector,
                             ucu::Vector)
 #objectif du problème avec pénalité lagrangienne sans contrainte :
 pen_mpcc=CreatePenaltyNLP(alas,xj,rho,usg,ush,uxl,uxu,ucl,ucu)

 #initialise notre problème pénalisé avec des contraintes actives w
 return ActifMPCCmod.MPCC_actif(pen_mpcc,alas.r,alas.s,alas.t,
                            alas.mod.nb_comp,alas.mod.paramset,
                            alas.mod.algoset.direction,
                            alas.mod.algoset.linesearch)
end

function CreatePenaltyNLP(alas::ALASMPCCmod.ALASMPCC,
                          xj::Vector,rho::Vector,
                          usg::Vector,ush::Vector,
                          uxl::Vector,uxu::Vector,
                          ucl::Vector,ucu::Vector)
 n=length(alas.mod.mp.meta.x0)
 nb_comp=alas.mod.nb_comp

 penf(x,yg,yh)=NLPModels.obj(alas.mod.mp,x)+Penaltygen(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 penf(x)=penf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 gpenf(x,yg,yh)=vec([NLPModels.grad(alas.mod.mp,x)' zeros(2*alas.mod.nb_comp)'])+GradPenaltygen(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 gpenf(x)=gpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 gfpenf(x,yg,yh)=ObjGradPenaltygen(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 gfpenf(x)=gfpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 Hpenf(x,yg,yh)=HessPenaltygen(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 Hpenf(x)=Hpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])


 return NLPModels.SimpleNLPModel(x->penf(x), xj,
                                lvar=[alas.mod.mp.meta.lvar;alas.tb*ones(2*alas.mod.nb_comp)], 
                                uvar=[alas.mod.mp.meta.uvar;Inf*ones(2*alas.mod.nb_comp)],
				g=x->gpenf(x),H=x->Hpenf(x),fg=x->gfpenf(x))
end

function UpdatePenaltyNLP(alas::ALASMPCCmod.ALASMPCC,
                          rho::Vector,xj::Vector,
                          usg::Vector,ush::Vector,
                          uxl::Vector,uxu::Vector,
                          ucl::Vector,ucu::Vector,
                          pen_nlp::NLPModels.AbstractNLPModel;gradpen::Vector=[],objpen::Float64=zeros(0),crho::Float64=1.0)
 n=length(alas.mod.mp.meta.x0)

 penf(x,yg,yh)=NLPModels.obj(alas.mod.mp,x)+Penaltygen(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 penf(x)=penf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 gpenf(x,yg,yh)=vec([NLPModels.grad(alas.mod.mp,x)' zeros(2*alas.mod.nb_comp)'])+GradPenaltygen(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 gpenf(x)=gpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 gfpenf(x,yg,yh)=ObjGradPenaltygen(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 gfpenf(x)=gfpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 Hpenf(x,yg,yh)=HessPenaltygen(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 Hpenf(x)=Hpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 pen_nlp.f=x->penf(x)
 pen_nlp.g=x->gpenf(x)
 pen_nlp.fg=x->gfpenf(x)
 pen_nlp.H=x->Hpenf(x)

 if isempty(objpen)
  return pen_nlp
 else
   if minimum(rho)==maximum(rho)
    #if minimum(rho)==alas.mod.paramset.rhomax
    #fobj=NLPModels.obj(alas.mod.mp,xj)
    #f=fobj+alas.mod.paramset.rho_update*(objpen-fobj)
    #g=alas.mod.paramset.rho_max*crho*(gradpen-vec([NLPModels.grad(alas.mod.mp,xj)' zeros(2*alas.mod.nb_comp)']))
    f,g=gfpenf(xj)
   else
    f,g=gfpenf(xj)
   end

  return pen_nlp,f,g
 end
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
function RhoUpdate(alas::ALASMPCC,rho::Vector,crho::Float64,err::Vector;l::Int64=0)
 #rho[find(x->x>0,err)]*=alas.rho_update
 #rho*=alas.mod.paramset.rho_update
 rho=min.(rho*alas.mod.paramset.rho_update,ones(length(rho))*alas.mod.paramset.rho_max*crho)

 return rho
end

"""
Renvoie : rho_eq, rho_ineq_var, rho_ineq_cons
"""
function RhoDetail(alas::ALASMPCC,rho::Vector)
 nb_ineq_lvar=length(alas.mod.mp.meta.lvar)
 nb_ineq_uvar=length(alas.mod.mp.meta.uvar)
 nb_ineq_lcons=length(alas.mod.mp.meta.lcon)
 nb_ineq_ucons=length(alas.mod.mp.meta.ucon)
 nb_comp=alas.mod.nb_comp

 return rho[1:nb_comp],rho[nb_comp+1:2*nb_comp],rho[2*nb_comp+1:2*nb_comp+nb_ineq_lvar],
        rho[2*nb_comp+nb_ineq_lvar+1:2*nb_comp+nb_ineq_lvar+nb_ineq_uvar],
        rho[2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+1:2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+nb_ineq_lcons],
        rho[2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+nb_ineq_lcons+1:2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+nb_ineq_lcons+nb_ineq_ucons]
end

#end of module
end
