module Stopping1Dmod

import NLPModels.AbstractNLPModel
import ActifModelmod.grad

type Stopping1D

    #Problem tolerances
    atol :: Float64                  # absolute tolerance
    rtol :: Float64                  # relative tolerance
    unbounded_threshold :: Float64   # below this value, the problem is declared unbounded

    tau_wolfe      :: Float64
    tau_armijo     :: Float64
    ite_max_armijo :: Int
    ite_max_wolfe  :: Int
    wolfe_update   :: Float64
    armijo_update  :: Float64
    hz_fact        :: Float64
    hz_prec        :: Float64

    # fine grain control on ressources
    max_obj_f           :: Int       # max objective function (f) evaluations allowed
    max_obj_grad        :: Int       # max objective gradient (g) evaluations allowed
    max_obj_hess        :: Int       # max objective hessian (H) evaluations allowed
    max_obj_hv          :: Int       # max objective H*v (HV) evaluations allowed

    # global control on ressources
    max_eval            :: Int       # max evaluations (f+g+H+Hv) allowed
    max_iter            :: Int       # max iterations allowed
    max_time            :: Float64   # max elapsed time allowed

    # global information to the stopping manager
    start_time          :: Float64   # starting time of the execution of the method

    #result
    wolfe_step          :: Bool
    iter_wolfe          :: Int64
    armijo_step         :: Bool
    iter_armijo         :: Int64
    hz_step             :: Bool

    # Stopping properties
    unbounded           :: Bool
    tired               :: Bool

    function Stopping1D(;atol               :: Float64  = 1.0e-8,
                        rtol                :: Float64  = 1.0e-6,
                        unbounded_threshold :: Float64  = -1.0e50,
                        tau_wolfe           :: Float64  = 0.6,
                        tau_armijo          :: Float64  = 0.1,
                        ite_max_armijo      :: Int      = 100,
                        ite_max_wolfe       :: Int      = 10,
                        wolfe_update        :: Float64  = 5.0,
                        armijo_update       :: Float64  = 0.9,
                        hz_fact             :: Float64  = -0.8,
                        hz_prec             :: Float64  = 1e-10,
                        max_obj_f           :: Int      = typemax(Int),
                        max_obj_grad        :: Int      = typemax(Int),
                        max_obj_hess        :: Int      = typemax(Int),
                        max_obj_hv          :: Int      = typemax(Int),
                        max_eval            :: Int      = 20000,
                        max_iter            :: Int      = 5000,
                        max_time            :: Float64  = 600.0,
                        optimality_residual :: Function = x -> norm(x,Inf),
                        kwargs...)
        
        return new(atol, rtol, unbounded_threshold,
                   tau_wolfe,tau_armijo,
                   ite_max_armijo, ite_max_wolfe,
                   wolfe_update, armijo_update,
                   hz_fact,hz_prec,
                   max_obj_f, max_obj_grad, max_obj_hess, max_obj_hv, max_eval,
                   max_iter, max_time, NaN, false,0,
                   false,0,false,false,false)
    end
end

function start!(nlp    :: AbstractNLPModel,
                s      :: Stopping1D,
                x      :: Array{Float64,1},
                hg     :: Float64,
                ht     :: Float64,
                slope  :: Float64,
                step   :: Float64,
                d      :: Array{Float64,1},
                gradft :: Array{Float64,1})

 s.iter_armijo = 0
 s.tired = s.iter_armijo >= s.ite_max_armijo

 s.armijo_step = _armijo_stop!(nlp, s, x, hg, ht, slope, step)

 OK = s.armijo_step || s.tired

 return OK
end

function hz_start!(nlp    :: AbstractNLPModel,
                   s      :: Stopping1D,
                   x      :: Array{Float64,1},
                   hg     :: Float64,
                   ht     :: Float64,
                   slope  :: Float64,
                   step   :: Float64,
                   d      :: Array{Float64,1},
                   gradft :: Array{Float64,1})

 s.iter_armijo = 0
 s.tired = s.iter_armijo >= s.ite_max_armijo

 s.armijo_step = _armijo_stop!(nlp, s, x, hg, ht, slope, step)

 slope_t = dot(d, gradft)

 s.hz_step = (ht <= hg + s.hz_prec*abs(hg)) && (slope_t <= s.hz_fact * slope)

 OK = s.armijo_step || s.hz_step || s.tired

 return OK
end

function aw_start!(nlp    :: AbstractNLPModel,
                   s      :: Stopping1D,
                   x      :: Array{Float64,1},
                   hg     :: Float64,
                   ht     :: Float64,
                   slope  :: Float64,
                   step   :: Float64,
                   d      :: Array{Float64,1},
                   gradft :: Array{Float64,1})

 s.iter_wolfe = 0
 s.tired = s.iter_wolfe >= s.ite_max_wolfe

 s.armijo_step = _armijo_stop!(nlp, s, x, hg, ht, slope, step)
 s.wolfe_step  = _wolfe_stop!(nlp, s, x, slope, d, gradft)

 OK = (s.armijo_step && s.wolfe_step) || s.tired

 return OK
end

function stop!(nlp    :: AbstractNLPModel,
               s      :: Stopping1D,
               x      :: Array{Float64,1},
               hg     :: Float64,
               ht     :: Float64,
               slope  :: Float64,
               step   :: Float64,
               d      :: Array{Float64,1},
               gradft :: Array{Float64,1})

 s.iter_armijo += 1
 s.tired = s.iter_armijo >= s.ite_max_armijo
 s.armijo_step = _armijo_stop!(nlp, s, x, hg, ht, slope, step)

 s.unbounded = ht < s.unbounded_threshold || s.unbounded

 OK = s.armijo_step || s.tired || s.unbounded

 return OK
end

function hz_stop!(nlp    :: AbstractNLPModel,
                  s      :: Stopping1D,
                  x      :: Array{Float64,1},
                  hg     :: Float64,
                  ht     :: Float64,
                  slope  :: Float64,
                  step   :: Float64,
                  d      :: Array{Float64,1},
                  gradft :: Array{Float64,1})

 s.iter_armijo += 1
 s.tired = s.iter_armijo >= s.ite_max_armijo
 s.armijo_step = _armijo_stop!(nlp, s, x, hg, ht, slope, step)

 s.unbounded = ht < s.unbounded_threshold || s.unbounded

 OK = s.armijo_step || s.tired || s.unbounded

 #In case, we don't have an Armijo step
 good_grad=false

 if !OK && ht <= hg+s.hz_prec*abs(hg)

  gradft    = grad(nlp, x)
  slope_t   = dot(d,gradft)

  good_grad = true

  s.hz_step = slope_t <= s.hz_fact*slope
 end

 OK = OK || s.hz_step

 return OK, good_grad, gradft
end

function aw_stop!(nlp    :: AbstractNLPModel,
                  s      :: Stopping1D,
                  x      :: Array{Float64,1},
                  hg     :: Float64,
                  ht     :: Float64,
                  slope  :: Float64,
                  step   :: Float64,
                  d      :: Array{Float64,1},
                  gradft :: Array{Float64,1})

 s.iter_wolfe += 1
 s.tired = s.iter_wolfe >= s.ite_max_wolfe

 s.armijo_step = _armijo_stop!(nlp, s, x, hg, ht, slope, step)
 s.wolfe_step  = _wolfe_stop!(nlp, s, x, slope, d, gradft)

 s.unbounded = ht < s.unbounded_threshold || s.tired || s.unbounded

 OK = (s.armijo_step && s.wolfe_step) || s.tired

 return OK
end

############################################################################
#
# Armijo and Wolfe check private functions
#
############################################################################
function _wolfe_stop!(nlp    :: AbstractNLPModel,
                      s      :: Stopping1D,
                      x      :: Array{Float64,1},
                      slope  :: Float64,
                      d      :: Array{Float64,1},
                      gradft :: Array{Float64,1})

 slope_t = dot(d, gradft)

 s.wolfe_step = slope_t < s.tau_wolfe*slope

 OK = s.wolfe_step

 return OK
end

function _armijo_stop!(nlp   :: AbstractNLPModel,
                       s     :: Stopping1D,
                       x     :: Array{Float64,1}, 
                       hg    :: Float64, 
                       ht    :: Float64,
                       slope :: Float64, 
                       step  :: Float64)

 s.armijo_step = ht - hg <= s.tau_armijo*step*slope

 OK = s.armijo_step

 return OK
end

#end of module
end
