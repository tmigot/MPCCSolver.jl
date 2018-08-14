"""
Package de fonction pour la recherche linéaire
"""
module LineSearch

import ActifModelmod.obj, ActifModelmod.grad
import Stopping1Dmod.Stopping1D
import Stopping1Dmod.stop!, Stopping1Dmod.start!
import Stopping1Dmod.wolfe_stop!, Stopping1Dmod.armijo_stop!
import NLPModels.AbstractNLPModel

"""
Armijo : Backtracking line search
         1D Minimization
"""
function armijo(ma        :: AbstractNLPModel,
                stp       :: Stopping1D,
                xj        :: AbstractVector,
                d         :: AbstractVector,
                hg        :: Float64,
                stepmax   :: Float64,
                slope     :: Float64,
                old_alpha :: Float64;
                verbose   :: Bool=true, kwargs...)

 good_grad=false
 nbW=0
 nbk=0
 step=min(stepmax,1.0)

 tau_wolfe      = stp.tau_wolfe
 tau_armijo     = stp.tau_armijo
 armijo_update  = stp.armijo_update
 wolfe_update   = stp.wolfe_update
 ite_max_wolfe  = stp.ite_max_wolfe
 ite_max_armijo = stp.ite_max_armijo

 ht=obj(ma,xj+step*d)

 #critère d'Armijo : f(x+alpha*d)-f(x)<=tau_a*alpha*grad f^Td
 while nbk<ite_max_armijo && ht-hg>tau_armijo*step*slope
  step*=armijo_update #step=step_0*(1/2)^m
  ht=obj(ma,xj+step*d)
  nbk+=1
 end
 step=step

 return step,good_grad,ht,nbk,nbW,gradft
end

"""
ArmijoWolfe : Backtracking line search + amélioration si le pas initial peut être augmenté
         1D Minimization
"""
function armijo_wolfe(ma        :: AbstractNLPModel, #renommer le ma!!
                      stp       :: Stopping1D,
                      xj        :: AbstractVector,
                      d         :: AbstractVector,
                      hg        :: Float64,
                      stepmax   :: Float64,
                      slope     :: Float64,
                      old_alpha :: Float64;
                      verbose   :: Bool=true, kwargs...)

 good_grad=false
 nbW=0
 step=min(stepmax,1.0)

 tau_wolfe      = stp.tau_wolfe
 tau_armijo     = stp.tau_armijo
 armijo_update  = stp.armijo_update
 wolfe_update   = stp.wolfe_update
 ite_max_wolfe  = stp.ite_max_wolfe
 ite_max_armijo = stp.ite_max_armijo

 #First try to increase t to satisfy loose Wolfe condition 

 xjp = xj+step*d
 ht  = obj(ma, xjp)

 gradft  = grad(ma, xjp)

 slope_t = dot(d, gradft)

 OK = wolfe_stop!(ma, stp, xj, slope, d, gradft) && armijo_stop!(ma, stp, xj, hg, ht, slope, step)

 while OK && nbW < ite_max_wolfe && step<stepmax

  step    = min(step*wolfe_update,stepmax) #on ne peut pas dépasser le pas maximal
  xjp     = xj+step*d
  ht      = obj(ma, xjp)
  gradft  = grad(ma, xjp)
  slope_t = dot(d, gradft)

  OK = wolfe_stop!(ma, stp, xj, slope, d, gradft) && armijo_stop!(ma, stp, xj, hg, ht, slope, step)

  nbW+=1
 end

 #In this case the problem is very likely to be unbounded
 if nbW==ite_max_wolfe

  stp.unbounded = true

  return step,good_grad,ht,gradft
 end

 #critère d'Armijo : f(x+alpha*d)-f(x)<=tau_a*alpha*grad f^Td
 OK = start!(ma, stp, xj, hg, ht, slope, step, d, gradft)
 while !OK

  step *= armijo_update #step=step_0*(1/2)^m
  xjp   = xj+step*d
  ht    = obj(ma, xjp)

  OK = stop!(ma, stp, xj, hg, ht, slope, step, d, gradft)

 end

 if stp.iter_armijo == 0
  good_grad = true
 end

 return step,good_grad,ht,gradft
end

"""
ArmijoWolfe : 'Newarmijo_wolfe' de JPD
Problème avec Hager et Zhang numerical trick
+ Ca n'a pas passé mon test... même sans Hager et Zhang trick...
"""
function armijo_wolfe_hz(ma        :: AbstractNLPModel,
                         stp       :: Stopping1D,
                         xj        :: AbstractVector,
                         d         :: AbstractVector,
                         hg        :: Float64,
                         stepmax   :: Float64,
                         slope     :: Float64,
                         old_alpha :: Float64;
                         verbose   :: Bool=true, kwargs...)

 # Perform improved Armijo linesearch.
 nbk = 0
 nbW = 0
 step = min(stepmax,1.0)

 tau_wolfe      = stp.tau_wolfe
 tau_armijo     = stp.tau_armijo
 armijo_update  = stp.armijo_update
 wolfe_update   = stp.wolfe_update
 ite_max_wolfe  = stp.ite_max_wolfe
 ite_max_armijo = stp.ite_max_armijo

 #First try to increase t to satisfy loose Wolfe condition 
 ht=obj(ma,xj+step*d)
 gradft=grad(ma,xj+step*d)
 slope_t=dot(d,gradft)
 while (slope_t<tau_wolfe*slope) && (ht-hg<=tau_armijo*step*slope) && (nbW<ite_max_armijo) && step<stepmax
  step=min(step*wolfe_update,stepmax) #on ne peut pas dépasser le pas maximal
  ht=obj(ma,xj+step*d)
  slope_t=dot(d,grad(ma,xj+step*d))

  nbW+=1
 end

# Hager & Zhang numerical trick
 hgoal = hg+slope*step*tau_armijo
 fact=-0.8
 prec=1e-10
 Armijo = (ht <= hgoal) || ((ht <= hg + prec*abs(hg)) && (slope_t <= fact * slope))
 good_grad=true
 #critère d'Armijo : f(x+alpha*d)-f(x)<=tau_a*alpha*grad f^Td
 while nbk<ite_max_armijo && !Armijo
  step*=armijo_update #step=step_0*(1/2)^m
  ht=obj(ma,xj+step*d)
  hgoal = hg+slope*step*tau_armijo

  Armijo=false
  good_grad=false
  if ht <= hgoal
   Armijo=true
  elseif ht <= hg+prec*abs(hg)
   gradft=grad(ma,xj+step*d)
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
