"""
Package de fonction pour la recherche linéaire d'un MPCCActif

liste des fonctions :
Armijo(ma::ActifMPCCmod.MPCC_actif,xj::Any,d::Any,hg::Any,gradft::Any,stepmax::Float64,slope::Float64;
                verbose :: Bool=false, kwargs...)
ArmijoWolfe(ma::ActifMPCCmod.MPCC_actif,xj::Any,d::Any,hg::Any,gradft::Any,stepmax::Float64,
                slope::Float64;verbose :: Bool=false, kwargs...)
"""
module LineSearch

using ActifMPCCmod

"""
Armijo : Backtracking line search
         1D Minimization
"""
function Armijo(ma::ActifMPCCmod.MPCC_actif,xj::Any,d::Any,hg::Any,stepmax::Float64,slope::Float64,old_alpha::Float64;
                verbose :: Bool=false, kwargs...)

 good_grad=false
 nbW=0
 nbk=0

 step=min(stepmax,1.0)

 ht=ActifMPCCmod.obj(ma,xj+step*d)

 #critère d'Armijo : f(x+alpha*d)-f(x)<=tau_a*alpha*grad f^Td
 while nbk<ma.paramset.ite_max_armijo && ht-hg>ma.paramset.tau_armijo*step*slope
  step*=ma.paramset.armijo_update #step=step_0*(1/2)^m
  ht=ActifMPCCmod.obj(ma,xj+step*d)
  nbk+=1
 end
 step=step

 return step,good_grad,ht,nbk,nbW,gradft
end

"""
ArmijoWolfe : Backtracking line search + amélioration si le pas initial peut être augmenté
         1D Minimization
"""
function ArmijoWolfe(ma::ActifMPCCmod.MPCC_actif,xj::AbstractVector,d::AbstractVector,hg::Float64,stepmax::Float64,slope::Float64,old_alpha::Float64;
                verbose :: Bool=true, kwargs...)

 good_grad=false
 nbW=0
 nbk=0
 step=min(stepmax,1.0)

 #First try to increase t to satisfy loose Wolfe condition 
 xjp=xj+step*d
 ht=ActifMPCCmod.obj(ma,xjp)

 gradft=ActifMPCCmod.grad(ma,xjp)
 slope_t=dot(d,gradft)

 while (slope_t<ma.paramset.tau_wolfe*slope) && (ht-hg<=ma.paramset.tau_armijo*step*slope) && (nbW<ma.paramset.ite_max_wolfe) && step<stepmax

  step=min(step*ma.paramset.wolfe_update,stepmax) #on ne peut pas dépasser le pas maximal
  xjp=xj+step*d
  ht=ActifMPCCmod.obj(ma,xjp)
  gradft=ActifMPCCmod.grad(ma,xjp)
  slope_t=dot(d,gradft)

  nbW+=1
 end

 #In this case the problem is very likely to be unbounded
 if nbW==ma.paramset.ite_max_wolfe
@show step stepmax
  return step,good_grad,ht,nbk,nbW,gradft
 end

 #critère d'Armijo : f(x+alpha*d)-f(x)<=tau_a*alpha*grad f^Td
 while nbk<ma.paramset.ite_max_armijo && ht-hg>ma.paramset.tau_armijo*step*slope
  step*=ma.paramset.armijo_update #step=step_0*(1/2)^m
  xjp=xj+step*d
  ht=ActifMPCCmod.obj(ma,xjp)

  nbk+=1
 end

 if nbk==0
  good_grad=true
 end

 return step,good_grad,ht,nbk,nbW,gradft
end

"""
ArmijoWolfe : 'Newarmijo_wolfe' de JPD
Problème avec Hager et Zhang numerical trick
+ Ca n'a pas passé mon test... même sans Hager et Zhang trick...
"""
function ArmijoWolfeHZ(ma::ActifMPCCmod.MPCC_actif,xj::Any,d::Any,hg::Any,stepmax::Float64,
                slope::Float64,old_alpha::Float64;verbose :: Bool=false, kwargs...)

 # Perform improved Armijo linesearch.
 nbk = 0
 nbW = 0
 step = min(stepmax,1.0)

 #First try to increase t to satisfy loose Wolfe condition 
 ht=ActifMPCCmod.obj(ma,xj+step*d)
 gradft=ActifMPCCmod.grad(ma,xj+step*d)
 slope_t=dot(d,gradft)
 while (slope_t<ma.paramset.tau_wolfe*slope) && (ht-hg<=ma.paramset.tau_armijo*step*slope) && (nbW<ma.paramset.ite_max_armijo) && step<stepmax
  step=min(step*ma.paramset.wolfe_update,stepmax) #on ne peut pas dépasser le pas maximal
  ht=ActifMPCCmod.obj(ma,xj+step*d)
  slope_t=dot(d,ActifMPCCmod.grad(ma,xj+step*d))

  nbW+=1
 end

# Hager & Zhang numerical trick
 hgoal = hg+slope*step*ma.paramset.tau_armijo
 fact=-0.8
 prec=1e-10
 Armijo = (ht <= hgoal) || ((ht <= hg + prec*abs(hg)) && (slope_t <= fact * slope))
 good_grad=true
 #critère d'Armijo : f(x+alpha*d)-f(x)<=tau_a*alpha*grad f^Td
 while nbk<ma.paramset.ite_max_armijo && !Armijo
  step*=ma.paramset.armijo_update #step=step_0*(1/2)^m
  ht=ActifMPCCmod.obj(ma,xj+step*d)
  hgoal = hg+slope*step*ma.paramset.tau_armijo

  Armijo=false
  good_grad=false
  if ht <= hgoal
   Armijo=true
  elseif ht <= hg+prec*abs(hg)
   gradft=ActifMPCCmod.grad(ma,xj+step*d)
   slope_t = dot(d,gradft)
   good_grad=true
   if slope_t <= fact*slope
    Armijo = true
   end
  end

  nbk+=1
 end

 good_grad=false #à enlever ?
 return step,good_grad,ht,nbk,nbW,gradft
end

#end of module
end
