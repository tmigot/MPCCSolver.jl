module UncstrndSolvemod

import ActifModelmod.ActifModel
using ActifModelmod

import OutputLSmod.OutputLS
import RUncstrndmod.RUncstrnd, RUncstrndmod.runc_update!
import Stopping1Dmod.Stopping1D

using NLPModels

#package to solve unconstrained minimization pb
type UncstrndSolve

  nlp  :: ActifModel
  x    :: Vector
  runc :: RUncstrnd
  sunc :: Stopping1D

  func_1d :: Function

 function UncstrndSolve(nlp :: ActifModel,
                        x :: Vector, 
                        runc :: RUncstrnd, 
                        sunc :: Stopping1D,
                        func_1d :: Function)

  return new(nlp,x,runc,sunc,func_1d)
 end

#end of type
end 

function set_x!(unc :: UncstrndSolve, x :: Vector)
 unc.x = x
 return unc
end

function solve_1d(unc :: UncstrndSolve; stepmax :: Float64 = Inf)
 x = unc.x
 d = unc.runc.d

 scale   = 1.0

 slope   = dot(unc.runc.∇f, d)
 step    = unc.runc.step

 hg = unc.runc.fx

 step,good_grad,ht,nbarmijo,nbwolfe,gradft=unc.func_1d(unc.nlp,unc.sunc,
                                                       x, d,
                                                       hg,
                                                       stepmax,
                                                       scale*slope,
                                                       step)

 step *= scale
 xp = x + step*d

 beta=NaN
 ols = OutputLS(stepmax, step, slope, beta, nbarmijo, nbwolfe)

 runc_update!(unc.runc, x, step = step, d = d, ∇fp = gradft, fxp = ht)

 return xp, unc, ols, nbarmijo,nbwolfe,good_grad
end

#end of module
end
