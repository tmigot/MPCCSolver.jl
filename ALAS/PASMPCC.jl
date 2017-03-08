module PASMPCC

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
SlackComplementarityProjection(alas::ALASMPCC)
PenaltyFunc(alas::ALASMPCC,x::Vector,yg::Vector,yh::Vector,rho::Float64,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector)
LagrangeInit(alas::ALASMPCC,rho::Float64,xj::Vector)
LagrangeCompInit(alas::ALASMPCC,rho::Float64,xj::Vector)
LagrangeUpdate(alas::ALASMPCC,rho::Float64,xjk::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector,usg::Vector,ush::Vector)
solvePAS(alas::ALASMPCC)
"""

#TO DO List :
#Major :
# - rho pourrait devenir un vecteur de pénalité plutôt qu'une !?
# - centraliser les sorties d'observation dans une structure output
# - faire apparaitre le choix de la direction de descente dans les paramètres
#Minor :
# - changer l'ordre d'apparition des fonctions
# - paramètre pour décider ou non l'affichage de sortie dans solvePas
# - changer le nom du fichier en ALASMPCCmod

type ALASMPCC
 mod::MPCCmod.MPCC
 #paramètres du pb
 r::Float64
 s::Float64
 t::Float64

 #paramètres algorithmiques
 prec::Float64 #precision à 0
 ite_max::Int64 #entier >=0
 ite_max_viol::Int64 #entier >=0
 rho_init::Float64 #nombre >= 0
 rho_update::Float64 #nombre >=1
 #paramètres algorithmiques pour la minimisation
 tau_armijo::Float64 #entre (0,0.5)
 tau_wolfe::Float64 #entre (0.5,1)
end

function ALASMPCC(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64)
 return ALASMPCC(mod,r,s,t,mod.prec,40000,10,1.0,10.0,0.25,0.75)
end

function ALASMPCC(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64, prec::Float64)
 return ALASMPCC(mod,r,s,t,prec,40000,10,1.0,10.0,0.25,0.75)
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

 #projection sur les contraintes de positivité relaxés : yG>=-r et yH>=-r
 x[n+1:n+nb_comp]=max(x[n+1:n+nb_comp],-ones(nb_comp)*alas.r)
 x[n+nb_comp+1:n+2*nb_comp]=max(x[n+nb_comp+1:n+2*nb_comp],-ones(nb_comp)*alas.r)

 #projection sur les contraintes papillons : yG<=psi(yH,r,s,t) et yH<=psi(yG,r,s,t)
 for i=1:nb_comp
  if x[n+i]-Relaxationmod.psi(x[n+nb_comp+i],alas.r,alas.s,alas.t)>0 && x[n+nb_comp+i]-Relaxationmod.psi(x[n+i],alas.r,alas.s,alas.t)>0
   x[n+i]>=x[n+nb_comp+i] ? x[n+nb_comp+i]=Relaxationmod.psi(x[n+i],alas.r,alas.s,alas.t) : x[n+i]=Relaxationmod.psi(x[n+nb_comp+i],alas.r,alas.s,alas.t)
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
 Pen_ineq=norm(max(alas.mod.mp.meta.lvar-x+uxl/rho,0))^2+norm(max(x-alas.mod.mp.meta.uvar+uxu/rho,0))^2+norm(max(alas.mod.mp.meta.lcon-alas.mod.mp.c(x)+ucl/rho,0))^2+norm(max(alas.mod.mp.c(x)-alas.mod.mp.meta.ucon+ucu/rho,0))^2

 return alas.mod.mp.f(x)+Lagrangian+rho/2.0*(Pen_eq+Pen_ineq)
end

"""
Initialisation des multiplicateurs de Lagrange
"""
function LagrangeInit(alas::ALASMPCC,rho::Float64,xj::Vector)

  n=length(alas.mod.mp.meta.x0)
  uxl=max(rho*(alas.mod.mp.meta.lvar-xj[1:n]),zeros(n))
  uxu=max(rho*(xj[1:n]-alas.mod.mp.meta.uvar),zeros(n))
  ucl=max(rho*(alas.mod.mp.meta.lcon-alas.mod.mp.c(xj[1:n])),zeros(length(alas.mod.mp.meta.y0)))
  ucu=max(rho*(alas.mod.mp.c(xj[1:n])-alas.mod.mp.meta.ucon),zeros(length(alas.mod.mp.meta.y0)))
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
 
 lphi=max((xj[n+1:n+nb_comp]-Relaxationmod.psi(xj[n+1+nb_comp:n+2*nb_comp],alas.r,alas.s,alas.t)).*(xj[n+1+nb_comp:n+2*nb_comp]-Relaxationmod.psi(xj[n+1:n+nb_comp],alas.r,alas.s,alas.t)),zeros(nb_comp))
 lg=max(-rho*(xj[n+1:n+nb_comp]+alas.r),zeros(nb_comp))
 lh=max(-rho*(xj[n+1+nb_comp:n+2*nb_comp]+alas.r),zeros(nb_comp))

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
rho_update=5
 obj_viol=0.25

 beta=0.0 #specifique CG

 n=length(alas.mod.mp.meta.x0)

# S0 : initialisation du problème avec slack (projection sur _|_ )
 xj=SlackComplementarityProjection(alas)

#Tableau de sortie :
 s_xtab=collect(xj)

 #variables globales en sortie du LineSearch
 step=0.0
 outputArmijo=0
 wmax=[]

 #Initialisation multiplicateurs de la contrainte de complémentarité
 lg,lh,lphi=LagrangeCompInit(alas,rho,xj)

 # Initialisation multiplicateurs : uxl,uxu,ucl,ucu :
 uxl,uxu,ucl,ucu,usg,ush=LagrangeInit(alas,rho,xj)

# S1 : Initialisation du MPCC_Actif 
 #objectif du problème avec pénalité lagrangienne sans contrainte :
 penf(x,yg,yh)=PenaltyFunc(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 #nlp pénalisé
 pen_mpcc = NLPModels.ADNLPModel(x->penf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp]), xj, lvar=[alas.mod.mp.meta.lvar;-alas.r*ones(2*alas.mod.nb_comp)], uvar=[alas.mod.mp.meta.uvar;Inf*ones(2*alas.mod.nb_comp)])
 #initialise notre problème pénalisé avec des contraintes actives w
 ma=ActifMPCCmod.MPCC_actif(pen_mpcc,alas.r,alas.s,alas.t,length(alas.mod.G(alas.mod.mp.meta.x0)))

 # S2 : Major Loop Activation de contrainte
 #initialisation
 k=0
 xjk=xj
 move=true
 gradpen=NLPModels.grad(pen_mpcc,xj)
 #direction initial
 dj=zeros(n+2*alas.mod.nb_comp)

#println("start k=0,l=0 :", xjk)
 while k<k_max && (findfirst(x->x<0,[lg;lh;lphi])!=0 || MPCCmod.viol_contrainte_norm(alas.mod,xjk[1:n],xjk[n+1:n+alas.mod.nb_comp],xjk[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])>alas.prec || norm(gradpen[1:ma.n],Inf)>alas.prec) && outputArmijo==0 && move #pourquoi ma.n alors qu'on a n ?

println(" -- k: ",k," rho: ", rho,"| l : ", [lg;lh;lphi]," | gradf<e ", gradpen[1:ma.n], " | obj ",alas.mod.mp.f(xjk[1:n]))
  l=0
  xjkl=xjk

  #boucle pour augmenter le paramètre de pénalisation tant qu'on a pas baissé suffisament la violation
  while l<l_max && (l==0 || (k!=0 && (MPCCmod.viol_contrainte_norm(alas.mod,xjkl[1:n],xjkl[n+1:n+alas.mod.nb_comp],xjkl[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])>obj_viol*MPCCmod.viol_contrainte_norm(alas.mod,xjk[1:n],xjk[n+1:n+alas.mod.nb_comp],xjk[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp]) && MPCCmod.viol_contrainte_norm(alas.mod,xjk[1:n],xjk[n+1:n+alas.mod.nb_comp],xjk[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])>alas.prec ))) && outputArmijo==0

   #On ne met pas à jour rho si tout se passe bien.
   rho=l!=0?rho*rho_update : rho

   #mise à jour de la fonction objectif quand on met rho à jour
   penf(x,yg,yh)=PenaltyFunc(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
   ActifMPCCmod.setf(ma,x->penf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp]),xjk)

   #Unconstrained Solver modifié pour résoudre le sous-problème avec contraintes de bornes
   xjkl,ma.w,dj,step,wmax,outputArmijo,beta=UnconstrainedMPCCActif.LineSearchSolve(ma,xjkl,beta,dj) #specifique CG
   #xjkl,ma.w,dj,step,wmax,outputArmijo,beta=UnconstrainedMPCCActif.LineSearchSolve(ma,xjk,beta,dj) #on ne garde pas l'étape refusé.

#println("l: ",l," xjkl: ",xjkl, "dj ",dj, " pas ",step, " obj ",alas.mod.mp.f(xjkl[1:n]))

   move=norm(dj,Inf)<=alas.prec?false:true
   l+=1
   l>=l_max && println("Max itération en l (rho update)")
  end
  xjk=xjkl
println("xjk:",xjk)

  #Mise à jour des paramètres de la pénalité Lagrangienne (uxl,uxu,ucl,ucu)
  uxl,uxu,ucl,ucu,usg,ush=LagrangeUpdate(alas,rho,xjk,uxl,uxu,ucl,ucu,usg,ush)

  #met à jouer la fonction objectif après la mise à jour des multiplicateurs
  penf(x,yg,yh)=PenaltyFunc(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
  ActifMPCCmod.setf(ma,x->penf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp]),xjk)

  #Mise à jour des multiplicateurs de la complémentarité
  gradpen=NLPModels.grad(pen_mpcc,xjk)

  lg,lh,lphi=ActifMPCCmod.LSQComputationMultiplier(ma,gradpen,xjk)

  #Relaxation rule si on a fait un pas de Armijo-Wolfe et si un multiplicateur est du mauvais signe
  if k!=0 && dot(gradpen,dj)>=alas.tau_wolfe*dot(NLPModels.grad(pen_mpcc,xjk-step*dj),dj) && findfirst(x->x<0,[lg;lh;lphi])!=0
   ActifMPCCmod.RelaxationRule(ma,xjk,lg,lh,lphi,wmax)
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
 if MPCCmod.viol_contrainte_norm(alas.mod,xj[1:n],xj[n+1:n+alas.mod.nb_comp],xj[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])>alas.prec
  println("Failure : Infeasible Solution")
  stat=1
 end
 if norm(gradpen[1:ma.n],Inf)>alas.prec
  println("Failure : Non-optimal Solution")
  stat=1
 end

 return xj,stat,s_xtab,rho;
end

#end of module
end
