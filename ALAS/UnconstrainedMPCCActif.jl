"""
Package de fonction pour la minimisation ActifMPCC

liste des fonctions :
SteepestDescent(ma::ActifMPCCmod.MPCC_actif,xj::Vector)
Armijo(ma::ActifMPCCmod.MPCC_actif,xj::Any,d::Any,hg::Any,old_grad::Any,stepmax::Float64,slope::Float64;
                verbose :: Bool=false, kwargs...)
ArmijoWolfe(ma::ActifMPCCmod.MPCC_actif,xj::Any,d::Any,hg::Any,old_grad::Any,stepmax::Float64,
                slope::Float64;verbose :: Bool=false, kwargs...)
LineSearchSolve(ma::ActifMPCCmod.MPCC_actif,xj::Vector,beta::Float64,d::Any)
"""

module UnconstrainedMPCCActif

using ActifMPCCmod
using LineSearch
using DDirection

"""
Input :
ma : MPCC_actif
xj : vecteur initial
hd : direction précédente en version étendue
"""
function LineSearchSolve(ma::ActifMPCCmod.MPCC_actif,xj::Vector,hd::Any;scaling :: Bool = false,CG::Bool=true,direction::Function=DDirection.CGHZ)
#function LineSearchSolve(ma::ActifMPCCmod.MPCC_actif,xj::Vector,hd::Any;scaling :: Bool = false,CG::Bool=true,direction::Function=DDirection.NwtdirectionLDLt)
 output=0
 hd=ActifMPCCmod.redd(ma,hd)
 scale=1.0
 beta=ma.beta

 #xj est un vecteur de taille (n x length(bar_w))
 if length(xj) == (ma.n+2*ma.nb_comp)
  xj=vcat(xj[1:ma.n],xj[ma.n+ma.w13c],xj[ma.n+ma.nb_comp+ma.w24c])
 elseif length(xj) != (ma.n+length(ma.w13c)+length(ma.w24c))
  println("Error dimension : UnconstrainedSolve")
  return
 end

 #Choix de la direction :
 gradf=ActifMPCCmod.grad(ma,xj)

 #Newton
 #d=-inv(H)*gradf
 #H=H/norm(H)+0.01*eye(length(xj))
 #Calcul d'une direction de descente de taille (n + length(bar_w))
 d=direction(ma,gradf,xj,hd,beta)

 slope = dot(gradf,d)
 if slope > 0.0  # restart with negative gradient
  d = - gradf
  slope =  dot(gradf,d)
 end

 #Calcul du pas maximum (peut être infinie!)
 stepmax,wmax,wnew = ActifMPCCmod.PasMax(ma,xj,d)

 #Recherche linéaire
 old_grad=NaN #à définir quelque part...
 hg=ActifMPCCmod.obj(ma,xj)
 step,good_grad,ht,nbk,nbW=LineSearch.Armijo(ma,xj,d,hg,old_grad,stepmax,scale*slope)
 step*=scale

 xjp=xj+step*d
 good_grad || (gradft=ActifMPCCmod.grad(ma,xjp))

 sol = ActifMPCCmod.evalx(ma,xjp)
 dsol = ActifMPCCmod.evald(ma,d)
 s=xjp-xj
 y=gradft-gradf

 #MAJ de Beta : gradient conjugué
 #beta=direction(beta,gradft,gradf,y,d)
 #ma=ActifMPCCmod.setbeta(ma,beta) #à intégrer dans la fonction direction ?
 ma=direction(ma,beta,gradft,gradf,y,d,step)

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
  wnew = [] #si on a pas pris le stepmax alors aucune contrainte n'est ajouté
 end
 if nbk >= ma.ite_max 
  output=1
 end

 return sol,ma.w,dsol,step,wnew,output #on devrait aussi renvoyer le gradient
end

#end of module
end
