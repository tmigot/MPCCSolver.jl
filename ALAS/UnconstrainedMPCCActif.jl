"""
Package de fonction pour la minimisation ActifMPCC

liste des fonctions :
LineSearchSolve(ma::ActifMPCCmod.MPCC_actif,xj::Vector,hd::Any;scaling :: Bool = false)
"""

module UnconstrainedMPCCActif

using ActifMPCCmod
using NLPModels
using OutputLSmod

"""
Input :
ma : MPCC_actif
xj : vecteur initial
d : direction précédente en version étendue
"""

function LineSearchSolve(ma::ActifMPCCmod.MPCC_actif,
                         xj::Vector,d::Vector,
                         step::Float64,gradpen::Vector,hg::Float64;
                         scaling :: Bool = false)

 output=0
 d=ActifMPCCmod.redd(ma,d)
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

 gradf=ActifMPCCmod.grad(ma,xj,gradpen)
 gradft=Array{Float64,1}

 #Calcul d'une direction de descente de taille (n + length(bar_w))
 d=ma.direction(ma,gradf,xj,d,beta)

 slope = dot(gradf,d)

 if slope > 0.0  # restart with negative gradient
  d = - gradf
  slope =  dot(gradf,d)
 end

 #Calcul du pas maximum (peut être infinie!)
 stepmax,wmax,wnew = ActifMPCCmod.PasMax(ma,xj,d)

 #hg=ActifMPCCmod.obj(ma,xj)

 step,good_grad,ht,nbarmijo,nbwolfe=ma.linesearch(ma,xj,d,hg,gradft,stepmax,scale*slope,step)
 step*=scale

 ols=OutputLSmod.OutputLS(stepmax,step,slope,beta,nbarmijo,nbwolfe)

 xjp=xj+step*d

 sol = ActifMPCCmod.evalx(ma,xjp)
 dsol = ActifMPCCmod.evald(ma,d)

 #good_grad || (gradft=ActifMPCCmod.grad(ma,xjp))
 gradpen=NLPModels.grad(ma.nlp,sol)
 gradft=ActifMPCCmod.grad(ma,xjp,gradpen)

 s=xjp-xj
 y=gradft-gradf

 #MAJ des paramètres du calcul de la direction
 ma=ma.direction(ma,xjp,beta,gradft,gradf,y,d,step)

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
  wnew = zeros(Bool,0,0) #si on a pas pris le stepmax alors aucune contrainte n'est ajouté
 end
 if nbarmijo >= ma.paramset.ite_max_armijo || true in isnan.(sol) || true in isnan(ht)
  output=1
 end
 small_step=norm(xjp-xj,Inf)<=eps(Float64)?true:false #on fait un pas trop petit

 return sol,ma.w,dsol,step,wnew,output,small_step,ols,gradpen,ht 
end

#end of module
end
