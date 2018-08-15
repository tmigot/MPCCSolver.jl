"""
Package de fonction pour la recherche linéaire
"""
module LineSearch

import ActifModelmod.obj, ActifModelmod.grad
import NLPModels.AbstractNLPModel

import Stopping1Dmod.Stopping1D
import Stopping1Dmod.stop!, Stopping1Dmod.start!
import Stopping1Dmod.hz_start!, Stopping1Dmod.hz_stop!
import Stopping1Dmod.aw_start!, Stopping1Dmod.aw_stop!

################################################################################
#
# LineSearch générique
#
################################################################################
function gen_linesearch(nlp       :: AbstractNLPModel,
                        stp       :: Stopping1D,
                        xj        :: Vector, 
                        d         :: Vector;
                        hg        :: Float64 = NaN,
                        stepmax   :: Float64 = Inf,
                        slope     :: Float64 = NaN,
                        old_alpha :: Float64 = NaN,
                        verbose   :: Bool = false,
                        kwargs...)

 #Step 1: remplir les champs optionnels
 hg = isnan(hg) ? obj(nlp, xj) : hg
 if isnan(slope)
  gradf = grad(nlp, xj)
  slope = dot(d,gradf)
 end
 #old_alpha = ??

 #Step 2: procédure qui augmente le pas
 # ****

 #Step 3: procédure qui diminue le pas
 # ****

 return nothing
end


"""
Armijo : Backtracking line search
         1D Minimization
"""
function armijo(nlp       :: AbstractNLPModel,
                stp       :: Stopping1D,
                xj        :: AbstractVector,
                d         :: AbstractVector,
                hg        :: Float64,
                stepmax   :: Float64,
                slope     :: Float64,
                old_alpha :: Float64;
                verbose   :: Bool=true, kwargs...)

 step=min(stepmax,1.0)
 
 xjp = xj+step*d
 ht  = obj(nlp, xjp)

 gradft = Float64[]
 good_grad=false

 #critère d'Armijo : f(x+alpha*d)-f(x)<=tau_a*alpha*grad f^Td
 OK = start!(nlp, stp, xjp, hg, ht, slope, step, d, gradft)
 while !OK

  step *= stp.armijo_update #step=step_0*(1/2)^m
  xjp   = xj+step*d
  ht    = obj(nlp, xjp)

  OK = stop!(nlp, stp, xjp, hg, ht, slope, step, d, gradft)

 end

 return step,good_grad,ht,gradft
end

"""
ArmijoWolfe : Backtracking line search + amélioration si le pas initial peut être augmenté
         1D Minimization
"""
function armijo_wolfe(nlp       :: AbstractNLPModel,
                      stp       :: Stopping1D,
                      xj        :: AbstractVector,
                      d         :: AbstractVector,
                      hg        :: Float64,
                      stepmax   :: Float64,
                      slope     :: Float64,
                      old_alpha :: Float64;
                      verbose   :: Bool=true, kwargs...)

 #I.Initialization
 good_grad=false
 step = min(stepmax,1.0)
 xjp = xj+step*d
 ht  = obj(nlp, xjp)
 gradft  = grad(nlp, xjp)

 #First try to increase t to satisfy loose Wolfe condition 
 OK = aw_start!(nlp, stp, xjp, hg, ht, slope, step, d, gradft)
 while OK && step<stepmax

  step    = min(step*stp.wolfe_update,stepmax)
  xjp     = xj+step*d
  ht      = obj(nlp, xjp)
  gradft  = grad(nlp, xjp)

  OK = aw_start!(nlp, stp, xjp, hg, ht, slope, step, d, gradft)
 end
 good_grad = true

 #critère d'Armijo : f(x+alpha*d)-f(x)<=tau_a*alpha*grad f^Td
 OK = start!(nlp, stp, xjp, hg, ht, slope, step, d, gradft)
 while !OK

  step *= stp.armijo_update #step=step_0*(1/2)^m
  xjp   = xj+step*d
  ht    = obj(nlp, xjp)

  OK = stop!(nlp, stp, xjp, hg, ht, slope, step, d, gradft)

 end

 return step,good_grad,ht,gradft
end

"""
ArmijoWolfe : 'Newarmijo_wolfe' de JPD
"""
 # Perform improved Armijo linesearch.
function armijo_wolfe_hz(nlp        :: AbstractNLPModel,
                         stp       :: Stopping1D,
                         xj        :: AbstractVector,
                         d         :: AbstractVector,
                         hg        :: Float64,
                         stepmax   :: Float64,
                         slope     :: Float64,
                         old_alpha :: Float64;
                         verbose   :: Bool=true, kwargs...)
 #I.Initialization
 step = min(stepmax,1.0)
 xjp = xj+step*d
 ht  = obj(nlp, xjp)
 gradft  = grad(nlp, xjp)

 #First try to increase t to satisfy loose Wolfe condition 
 OK = aw_start!(nlp, stp, xjp, hg, ht, slope, step, d, gradft)
 while OK && step<stepmax

  step    = min(step*stp.wolfe_update,stepmax)
  xjp     = xj+step*d
  ht      = obj(nlp, xjp)
  gradft  = grad(nlp, xjp)

  OK = aw_start!(nlp, stp, xjp, hg, ht, slope, step, d, gradft)
 end
 good_grad = true

 # Hager & Zhang numerical trick
 #critère d'Armijo : f(x+alpha*d)-f(x)<=tau_a*alpha*grad f^Td
 OK = hz_start!(nlp, stp, xjp, hg, ht, slope, step, d, gradft)
 while !OK

  step *= stp.armijo_update #step=step_0*(1/2)^m
  xjp   = xj+step*d
  ht    = obj(nlp, xjp)

  OK, good_grad, gradft = hz_stop!(nlp, stp, xjp, hg, ht, slope, step, d, gradft)
 end

 return step,good_grad,ht,gradft
end

############################################################################
#
# Armijo sub_problem
#
############################################################################

#end of module
end
