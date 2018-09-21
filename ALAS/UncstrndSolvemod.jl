module UncstrndSolvemod

import ActifModelmod.ActifModel
using ActifModelmod

import OutputLSmod.OutputLS
import RUncstrndmod.RUncstrnd, RUncstrndmod.runc_update!
import Stopping1Dmod.Stopping1D

using NLPModels
import NLPModels: AbstractNLPModel

#package to solve unconstrained minimization pb
type UncstrndSolve

  nlp  :: AbstractNLPModel
  x    :: Vector
  runc :: RUncstrnd
  sunc :: Stopping1D

  func_1d :: Function

 function UncstrndSolve(nlp :: AbstractNLPModel,
                        x :: Vector, 
                        runc :: RUncstrnd, 
                        sunc :: Stopping1D,
                        func_1d :: Function)

  return new(nlp,x,runc,sunc,func_1d)
 end

#end of type
end 

############################################################################
#
# Methods to update the UncstrndSolve
# set_x
#
############################################################################

function set_x!(unc :: UncstrndSolve, x :: Vector)
 unc.x = x
 return unc
end

############################################################################
#
# Appel à une fonction pour la minimisation 1d
#
############################################################################

function solve_1d(unc :: UncstrndSolve; stepmax :: Float64 = Inf)

 
 x = unc.x
 d = unc.runc.d

 scale   = 1.0
 slope   = dot(unc.runc.∇f, d)
 step    = unc.runc.step
 hg = unc.runc.fx

 #Call a 1-D minimization function
 # Minimizes f along the direction d.
 step,good_grad,ht,gradft=unc.func_1d(unc.nlp,unc.sunc,
                                      x, d, hg, stepmax,
                                      scale*slope, step)

 step *= scale
 xp    = x + step*d

 ols = OutputLS(stepmax, step, slope, NaN,-1,-1)

 runc_update!(unc.runc, x, step = step, d = d, ∇fp = gradft, fxp = ht)

  #Final rending
  unc.runc.solved = 0
  nan_check = true in isnan.(x) || true in isnan(unc.runc.fxp)
  if unc.sunc.tired || nan_check
   unc.runc.solved = 1
   unc.runc.step   = 0.0
  elseif unc.sunc.unbounded
   unc.runc.solved = 2
  end

 #Update the solution, if we succeed
 unc.x = unc.runc.solved == 0 || unc.runc.solved == 3 ? xp : unc.x

 return xp, unc, ols, good_grad
end

#end of module
end
