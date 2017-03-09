module ALASMPCCmod

using ActifMPCCmod
using UnconstrainedMPCCActif
using MPCCmod
using Relaxationmod

using NLPModels

export solvePAS

"""
Type ALASMPCC : 

liste des constructeurs :
ALASMPCC(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64)
ALASMPCC(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64, prec::Float64)

liste des méthodes :


liste des accesseurs :


liste des fonctions :
solvePAS(alas::ALASMPCC)

SlackComplementarityProjection(alas::ALASMPCC)
PenaltyFunc(alas::ALASMPCC,x::Vector,yg::Vector,yh::Vector,rho::Float64,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector)
LagrangeInit(alas::ALASMPCC,rho::Float64,xj::Vector)
LagrangeCompInit(alas::ALASMPCC,rho::Float64,xj::Vector)
LagrangeUpdate(alas::ALASMPCC,rho::Float64,xjk::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector,usg::Vector,ush::Vector)
"""

#TO DO List :
#Major :
# - rho pourrait devenir un vecteur de pénalité plutôt qu'une !?
# - centraliser les sorties d'observation dans une structure output
# - faire apparaitre le choix de la direction de descente dans les paramètres
# - variables d'écart sur les contraintes de pénalités
#Minor :
# - paramètre pour décider ou non l'affichage de sortie texte dans solvePas

type ALASMPCC
 mod::MPCCmod.MPCC
 #paramètres du pb
 r::Float64
 s::Float64
 t::Float64
 tb::Float64 #pour l'instant =-r

 #paramètres algorithmiques
 prec::Float64 #precision à 0
 ite_max::Int64 #entier >=0
 ite_max_viol::Int64 #entier >=0
 rho_init::Float64 #nombre >= 0
 rho_update::Float64 #nombre >=1
 goal_viol
 #paramètres algorithmiques pour la minimisation
 tau_armijo::Float64 #entre (0,0.5)
 tau_wolfe::Float64 #entre (tau_armijo,1)
end

function ALASMPCC(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64)
 return ALASMPCC(mod,r,s,t,-r,mod.prec)
end

function ALASMPCC(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64, prec::Float64)
 ite_max=5000
# ite_max=12
 ite_max_viol=20
 rho_init=1.0
 rho_update=2.0
 goal_viol=0.5
 tau_armijo=0.25
 tau_wolfe=0.9
 return ALASMPCC(mod,r,s,t,-r,prec,ite_max,ite_max_viol,rho_init,rho_update,goal_viol,tau_armijo,tau_wolfe)
end

"""
solvePAS() : méthode de penalisation/activation de contraintes DHKM 16'

Méthode avec gestion à l'intérieur du paramètre de pénalité
"""
function solvePAS(alas::ALASMPCC)

#Initialisation paramètres :
 rho=alas.rho_init
 k_max=alas.ite_max
 l_max=alas.ite_max_viol
 rho_update=alas.rho_update
 obj_viol=alas.goal_viol

 beta=0.0 #specifique CG

 n=length(alas.mod.mp.meta.x0)

# S0 : initialisation du problème avec slack (projection sur _|_ )
 xj=SlackComplementarityProjection(alas)

#Tableau de sortie :
 s_xtab=collect(xj)

 #variables globales en sortie du LineSearch
 step=0.0
 outputArmijo=0
 wnew=[]

 #Initialisation multiplicateurs de la contrainte de complémentarité
 lg,lh,lphi=LagrangeCompInit(alas,rho,xj)

 # Initialisation multiplicateurs : uxl,uxu,ucl,ucu :
 uxl,uxu,ucl,ucu,usg,ush=LagrangeInit(alas,rho,xj)

# S1 : Initialisation du MPCC_Actif 
 #objectif du problème avec pénalité lagrangienne sans contrainte :
 penf(x,yg,yh)=PenaltyFunc(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 penf(x)=penf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])
 #nlp pénalisé
 pen_mpcc = NLPModels.ADNLPModel(x->penf(x), xj,
                                lvar=[alas.mod.mp.meta.lvar;alas.tb*ones(2*alas.mod.nb_comp)], 
                                 uvar=[alas.mod.mp.meta.uvar;Inf*ones(2*alas.mod.nb_comp)])
 #initialise notre problème pénalisé avec des contraintes actives w
 ma=ActifMPCCmod.MPCC_actif(pen_mpcc,alas.r,alas.s,alas.t,length(alas.mod.G(alas.mod.mp.meta.x0)))

 # S2 : Major Loop Activation de contrainte
 #initialisation
 k=0
 xjk=xj
 small_step=false
 gradpen=NLPModels.grad(pen_mpcc,xj)
 #direction initial
 dj=zeros(n+2*alas.mod.nb_comp)

 l_negative=findfirst(x->x<0,[lg;lh;lphi])!=0
 feasible=MPCCmod.viol_contrainte_norm(alas.mod,xjk)<=alas.prec
 multi_norm=max(norm([usg;ush;uxl;uxu;ucl;ucu;lg;lh;lphi]),1)
 dual_feasible=norm(gradpen[1:n],Inf)/multi_norm<=alas.prec

println("start k=0,l=0 :", xjk)
 while k<k_max && (l_negative || !feasible || !dual_feasible) && outputArmijo==0 && !small_step

#println(" -- k: ",k," rho: ", rho,"| l : ", [lg;lh;lphi]," | gradf<e ", gradpen[1:ma.n], " | obj ",alas.mod.mp.f(xjk[1:n]))
  l=0
  xjkl=xjk

  #boucle pour augmenter le paramètre de pénalisation tant qu'on a pas baissé suffisament la violation
  while l<l_max && (l==0 || (k!=0 && (MPCCmod.viol_contrainte_norm(alas.mod,xjkl[1:n],xjkl[n+1:n+alas.mod.nb_comp],xjkl[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])>obj_viol*MPCCmod.viol_contrainte_norm(alas.mod,xjk[1:n],xjk[n+1:n+alas.mod.nb_comp],xjk[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp]) && MPCCmod.viol_contrainte_norm(alas.mod,xjk[1:n],xjk[n+1:n+alas.mod.nb_comp],xjk[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])>alas.prec ))) && outputArmijo==0 && !small_step #clarifier cette expression

   #On ne met pas à jour rho si tout se passe bien.
   rho=l!=0?rho*rho_update : rho

   #mise à jour de la fonction objectif quand on met rho à jour
   penf(x,yg,yh)=PenaltyFunc(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
   ActifMPCCmod.setf(ma,x->penf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp]),xjk)

   #Unconstrained Solver modifié pour résoudre le sous-problème avec contraintes de bornes
 xjkl,ma.w,dj,step,wnew,outputArmijo,beta=UnconstrainedMPCCActif.LineSearchSolve(ma,xjkl,beta,dj) #specifique CG 

   #small_step=step<=eps(Float64)?true:false #on fait un pas trop petit
   l+=1
   l>=l_max && println("Max itération en l (rho update). Iteration: ",k)
  end
  xjk=xjkl

  #Mise à jour des paramètres de la pénalité Lagrangienne (uxl,uxu,ucl,ucu)
  uxl,uxu,ucl,ucu,usg,ush=LagrangeUpdate(alas,rho,xjk,uxl,uxu,ucl,ucu,usg,ush)

  #met à jouer la fonction objectif après la mise à jour des multiplicateurs
  penf(x,yg,yh)=PenaltyFunc(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
  penf(x)=penf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])
  ActifMPCCmod.setf(ma,x->penf(x),xjk)

  #Mise à jour des multiplicateurs de la complémentarité
  gradpen=NLPModels.grad(pen_mpcc,xjk)

  lg,lh,lphi=ActifMPCCmod.LSQComputationMultiplier(ma,gradpen,xjk)
  l_negative=findfirst(x->x<0,[lg;lh;lphi])!=0 #vrai si un multiplicateur est negatif
  multi_norm=max(norm([usg;ush;uxl;uxu;ucl;ucu;lg;lh;lphi]),1)

  #feasible=MPCCmod.viol_contrainte_norm(alas.mod,xjk[1:n],xjk[n+1:n+alas.mod.nb_comp],xjk[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])<=alas.prec
  feasible=MPCCmod.viol_contrainte_norm(alas.mod,xjk)<=alas.prec
  dual_feasible=norm(gradpen[1:n],Inf)/multi_norm<=alas.prec
#println("k:",k," xjk:",xjk, " pas ",round(Int64,log(step))," rho:",round(Int64,log(rho))," |d|=",round(Int64,log(norm(dj)))," N=",round(Int64,100.0*norm(gradpen[1:n],Inf)/multi_norm)/100.0)

  if l_negative && (outputArmijo!=0 || small_step) #si on bloque mais qu'un multiplicateur est <0 on continue
   outputArmijo=0
   small_step=false
  end

  #Relaxation rule si on a fait un pas de Armijo-Wolfe et si un multiplicateur est du mauvais signe
  #quoi faire si on est bloqué ? i.e pas=0.0, on tente de relaxer
#  if k!=0 && dot(gradpen,dj)>=alas.tau_wolfe*dot(NLPModels.grad(pen_mpcc,xjk-step*dj),dj) && findfirst(x->x<0,[lg;lh;lphi])!=0
  if k!=0 && (dot(gradpen,dj)>=alas.tau_wolfe*dot(NLPModels.grad(pen_mpcc,xjk-step*dj),dj) || step==0.0) && findfirst(x->x<0,[lg;lh;lphi])!=0
   ActifMPCCmod.RelaxationRule(ma,xjk,lg,lh,lphi,wnew)
  end

  k+=1
  append!(s_xtab,collect(xjk))
  k>=k_max && println("Max itération Lagrangien augmenté")
 end

#Traitement finale :
 xj=xjk

 stat=0
 if outputArmijo!=0
  println("Failure : Armijo Failure")
  stat=1
 end
 if small_step
  println("Failure : too small step")
  stat=1
 end
 if MPCCmod.viol_contrainte_norm(alas.mod,xj[1:n],xj[n+1:n+alas.mod.nb_comp],xj[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])>alas.prec
  println("Failure : Infeasible Solution. norm: ",MPCCmod.viol_contrainte_norm(alas.mod,xjk[1:n],xjk[n+1:n+alas.mod.nb_comp],xjk[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp]))
  stat=1
 end
 if norm(gradpen[1:ma.n],Inf)>alas.prec
  if k>=k_max
   println("Failure : Non-optimal Sol. norm:",norm(gradpen[1:n],Inf))
   stat=1
  else
   println("Inexact : Fritz-John Sol. norm:",norm(gradpen[1:n],Inf))
   stat=0
  end
 end

 return xj,stat,s_xtab,rho;
end

"""
SlackComplementarityProjection : 
projete le point x0 sur la contrainte de complémentarité avec les slack
pré-requis : x0 doit être de taille n+2q
"""
function SlackComplementarityProjection(alas::ALASMPCC)

 #initialisation :
 x=[alas.mod.mp.meta.x0;alas.mod.G(alas.mod.mp.meta.x0);alas.mod.H(alas.mod.mp.meta.x0)]
 nb_comp=alas.mod.nb_comp
 n=length(x)-2*nb_comp

 #projection sur les contraintes de positivité relaxés : yG>=tb et yH>=tb
 x[n+1:n+nb_comp]=max(x[n+1:n+nb_comp],ones(nb_comp)*alas.tb)
 x[n+nb_comp+1:n+2*nb_comp]=max(x[n+nb_comp+1:n+2*nb_comp],ones(nb_comp)*alas.tb)

 #projection sur les contraintes papillons : yG<=psi(yH,r,s,t) et yH<=psi(yG,r,s,t)
 for i=1:nb_comp
  psiyg=Relaxationmod.psi(x[n+i],alas.r,alas.s,alas.t)
  psiyh=Relaxationmod.psi(x[n+nb_comp+i],alas.r,alas.s,alas.t)

  if x[n+i]-psiyh>0 && x[n+nb_comp+i]-psiyg>0
   x[n+i]>=x[n+nb_comp+i] ? x[n+nb_comp+i]=psiyg : x[n+i]=psiyh
  end
 end

 return x
end

"""
Fonction de pénalité utilisé
"""
function PenaltyFunc(alas::ALASMPCC,x::Vector,yg::Vector,yh::Vector,rho::Float64,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector)
 #mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64
 Lagrangian=dot(alas.mod.G(x)-yg,usg)+dot(alas.mod.H(x)-yh,ush)

 Pen_eq=norm(alas.mod.G(x)-yg)^2+norm(alas.mod.H(x)-yh)^2

 Pen_ineq_var=norm(max(alas.mod.mp.meta.lvar-x+uxl/rho,0))^2+norm(max(x-alas.mod.mp.meta.uvar+uxu/rho,0))^2

 Pen_ineq_cons=norm(max(alas.mod.mp.meta.lcon-alas.mod.mp.c(x)+ucl/rho,0))^2+norm(max(alas.mod.mp.c(x)-alas.mod.mp.meta.ucon+ucu/rho,0))^2

 return alas.mod.mp.f(x)+Lagrangian+rho/2.0*(Pen_eq+Pen_ineq_var+Pen_ineq_cons)
end

"""
Initialisation des multiplicateurs de Lagrange
"""
function LagrangeInit(alas::ALASMPCC,rho::Float64,xj::Vector)

  nc=length(alas.mod.mp.meta.y0) #nombre de contraintes
  n=length(alas.mod.mp.meta.x0)
  uxl=max(rho*(alas.mod.mp.meta.lvar-xj[1:n]),zeros(n))
  uxu=max(rho*(xj[1:n]-alas.mod.mp.meta.uvar),zeros(n))
  ucl=max(rho*(alas.mod.mp.meta.lcon-alas.mod.mp.c(xj[1:n])),zeros(nc))
  ucu=max(rho*(alas.mod.mp.c(xj[1:n])-alas.mod.mp.meta.ucon),zeros(nc))
  usg=zeros(alas.mod.nb_comp)
  ush=zeros(alas.mod.nb_comp)

 return uxl,uxu,ucl,ucu,usg,ush
end

"""
Initialisation des multiplicateurs de Lagrange pour la contrainte de complémentarité
"""
function LagrangeCompInit(alas::ALASMPCC,rho::Float64,xj::Vector)

 n=length(alas.mod.mp.meta.x0)
 nb_comp=alas.mod.nb_comp
 
 psiyg=Relaxationmod.psi(xj[n+1:n+nb_comp],alas.r,alas.s,alas.t)
 psiyh=Relaxationmod.psi(xj[n+1+nb_comp:n+2*nb_comp],alas.r,alas.s,alas.t)
 phi=(xj[n+1:n+nb_comp]-psiyh).*(xj[n+1+nb_comp:n+2*nb_comp]-psiyg)

 lphi=max(phi,zeros(nb_comp))
 lg=max(-rho*(xj[n+1:n+nb_comp]-alas.tb),zeros(nb_comp))
 lh=max(-rho*(xj[n+1+nb_comp:n+2*nb_comp]-alas.tb),zeros(nb_comp))

 return lg,lh,lphi
end

"""
Mise à jour des multiplicateurs de Lagrange
"""
function LagrangeUpdate(alas::ALASMPCC,rho::Float64,xjk::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector,usg::Vector,ush::Vector)

  n=length(alas.mod.mp.meta.x0)

  uxl=uxl+max(rho*(xjk[1:n]-alas.mod.mp.meta.uvar),-uxl)
  uxu=uxu+max(rho*(alas.mod.mp.meta.lvar-xjk[1:n]),-uxu)
  ucl=ucl+max(rho*(alas.mod.mp.c(xjk[1:n])-alas.mod.mp.meta.ucon),-ucl)
  ucu=ucu+max(rho*(alas.mod.mp.meta.lcon-alas.mod.mp.c(xjk[1:n])),-ucu)
  usg=usg+rho*(alas.mod.G(xjk[1:n])-xjk[n+1:n+alas.mod.nb_comp])
  ush=ush+rho*(alas.mod.H(xjk[1:n])-xjk[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 return uxl,uxu,ucl,ucu,usg,ush
end

#end of module
end
