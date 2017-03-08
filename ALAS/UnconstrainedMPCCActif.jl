"""
Package de fonction pour la minimisation ActifMPCC

liste des fonctions :
SteepestDescent(ma::ActifMPCCmod.MPCC_actif,xj::Vector)
Armijo(ma::ActifMPCCmod.MPCC_actif,xj::Any,d::Any,stepmax::Float64;
                tau_0::Float64=1.0e-4,tau_1::Float64=0.9999,
                bk_max::Int=50,nbWM :: Int=50, verbose :: Bool=false, kwargs...)
ArmijoWolfe(ma::ActifMPCCmod.MPCC_actif,xj::Any,d::Any,old_grad::Any,stepmax::Float64,slope::Float64;
                tau_0::Float64=1.0e-4,tau_1::Float64=0.9999,
                bk_max::Int=50,nbWM :: Int=50, verbose :: Bool=false, kwargs...)
LineSearchSolve(ma::ActifMPCCmod.MPCC_actif,xj::Vector,beta::Float64,d::Any)
"""

module UnconstrainedMPCCActif

using ActifMPCCmod

#TO DO List :
#Major:
# - recherche linéaire + sophistiquée (fonction ArmijoWolfe)
# - choix de la direction de descente (ajouter de nouvelles)

"""
SteepestDescent(ma::MPCC_actif,xj::Vector) : Calcul une direction de descente
"""
function SteepestDescent(ma::ActifMPCCmod.MPCC_actif,xj::Vector)
 return -ActifMPCCmod.grad(ma,xj)
end

"""
Armijo : appel de fonction compatible avec le 'Newarmijo_wolfe' de JPD
"""
function Armijo(ma::ActifMPCCmod.MPCC_actif,xj::Any,d::Any,old_grad::Any,stepmax::Float64,slope::Float64;
                tau_0::Float64=1.0e-4,tau_1::Float64=0.9999,
                bk_max::Int=50,nbWM :: Int=50, verbose :: Bool=false, kwargs...)

 good_grad=false
 nbW=0
 nbk=0
 step=min(stepmax,1.0)
 #slope = scale*dot(ActifMPCCmod.grad(ma,xj),d)

 hgoal=ActifMPCCmod.obj(ma,xj)
 ht=ActifMPCCmod.obj(ma,xj+step*d)

 #critère d'Armijo : f(x+alpha*d)-f(x)<=tau_a*alpha*grad f^Td
 while nbk<ma.ite_max && ht-hgoal>ma.tau_armijo*step*slope
  #step=step_0*(1/2)^m
  step*=0.8
  ht=ActifMPCCmod.obj(ma,xj+step*d)
  nbk+=1
 end

 return step,good_grad,ht,nbk,nbW
end

"""
EN TRAVAUX
Armijo : appel de fonction compatible avec le 'Newarmijo_wolfe' de JPD
"""
function ArmijoWolfe(ma::ActifMPCCmod.MPCC_actif,xj::Any,d::Any,old_grad::Any,stepmax::Float64,slope::Float64;
                tau_0::Float64=1.0e-4,tau_1::Float64=0.9999,
                bk_max::Int=50,nbWM :: Int=50, verbose :: Bool=false, kwargs...)

 # Perform improved Armijo linesearch.
 nbk = 0
 nbW = 0
 step = min(stepmax,1.0)
 good_grad=false
 ht=0.0

 return step,good_grad,ht,nbk,nbW
end

"""
La meme mais avec CG
Input :
ma : MPCC_actif
xj : vecteur initial
d : direction précédente (version étendue)
"""
function LineSearchSolve(ma::ActifMPCCmod.MPCC_actif,xj::Vector,beta::Float64,hd::Any;scaling :: Bool = false)

 output=0
 hd=ActifMPCCmod.redd(ma,hd)
 scale=1.0

 #xj est un vecteur de taille (n x length(bar_w))
 if length(xj) == (ma.n+2*ma.nb_comp)
  xj=vcat(xj[1:ma.n],xj[ma.n+ma.w13c],xj[ma.n+ma.nb_comp+ma.w24c])
 elseif length(xj) != (ma.n+length(ma.w13c)+length(ma.w24c))
  println("Error dimension : UnconstrainedSolve")
  return
 end

 gradf=ActifMPCCmod.grad(ma,xj)
 #Calcul d'une direction de descente de taille (n + length(bar_w))
 d = - gradf + beta*hd

 slope = dot(gradf,d)
 if slope > 0.0  # restart with negative gradient
  d = - gradf
  slope =  dot(gradf,d)
 end

 #Calcul du pas maximum (peut être infinie!)
 stepmax,wmax = ActifMPCCmod.PasMax(ma,xj,d)

 #Recherche linéaire
 old_grad=NaN #à définir quelque part...
 step,good_grad,ht,nbk,nbW=Armijo(ma,xj,d,old_grad,stepmax,scale*slope)
 step*=scale

 xjp=xj+step*d
 good_grad || (gradft=ActifMPCCmod.grad(ma,xjp))
 #gradft=ActifMPCCmod.grad(ma,xjp)
 sol = ActifMPCCmod.evalx(ma,xjp)
 dsol = ActifMPCCmod.evald(ma,d)
 s=xjp-xj
 y=gradft-gradf

 #MAJ de Beta
 #beta=0.0
 if dot(gradft,gradf)<0.2*dot(gradft,gradft) # Powell restart
#Intégrer dans une nouvelle fonction ?
  #Formula FR
  #β = (∇ft⋅∇ft)/(∇f⋅∇f) FR
  #beta=dot(gradft,gradft)/dot(gradf,gradf)
  #Formula PR
  #β = (∇ft⋅y)/(∇f⋅∇f)
  #beta=dot(gradft,y)/dot(gradf,gradf)
  #Formula HS
  #β = (∇ft⋅y)/(d⋅y)
  beta=dot(gradft,y)/dot(d,y)
  #Formula HZ
  n2y = dot(y,y)
  b1 = dot(y,d)
  #β = ((y-2*d*n2y/β1)⋅∇ft)/β1
  #beta = dot(y-2*d*n2y/b1,gradft)/b1
 end

 #MAJ du scaling
 if scaling
  scale=dot(y,s)/dot(y,y)
 end
 if scale <= 0.0
  scale=1.0
 end

 #si alpha=pas maximum alors on met w à jour.
 if stepmax == step
  ActifMPCCmod.setw(ma,wmax)
 else
  wmax = []
 end
 if nbk >= ma.ite_max 
  output=1
 end

 return sol,ma.w,dsol,step,wmax,output,beta #on devrait aussi renvoyer le gradient
end

#end of module
end
