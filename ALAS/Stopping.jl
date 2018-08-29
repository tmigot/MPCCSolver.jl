module Stopping

using NLPModels

import Stopping1Dmod.Stopping1D
import RUncstrndmod.RUncstrnd

# A stopping manager for iterative solvers
export TStopping, start!, stop

type TStopping

    atol :: Float64                  # absolute tolerance
    rtol :: Float64                  # relative tolerance
    unbounded_threshold :: Float64   # below this value, the problem is declared unbounded

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
    optimality0         :: Float64   # value of the optimality residual at starting point
    optimality_residual :: Function  # function to compute the optimality residual

    #active constraints feasible
    actfeas             :: Bool

    #result
    wolfe_step          :: Bool
    tau_wolfe           :: Float64
    iter                :: Int64
    sub_pb_solved       :: Bool

    # Stopping properties
    optimality          :: Float64
    optimal             :: Bool
    unbounded           :: Bool
    tired               :: Bool

    function TStopping(;atol                :: Float64  = 1.0e-8,
                        rtol                :: Float64  = 1.0e-6,
                        unbounded_threshold :: Float64  = -1.0e50,
                        max_obj_f           :: Int      = typemax(Int),
                        max_obj_grad        :: Int      = typemax(Int),
                        max_obj_hess        :: Int      = typemax(Int),
                        max_obj_hv          :: Int      = typemax(Int),
                        max_eval            :: Int      = 20000,
                        max_iter            :: Int      = 5000,
                        max_time            :: Float64  = 600.0,
                        optimality_residual :: Function = x -> norm(x,Inf),
                        wolfe_step          :: Bool     = true,
                        tau_wolfe           :: Float64  = 0.6,
                        kwargs...)
        
        return new(atol, rtol, unbounded_threshold,
                   max_obj_f, max_obj_grad, max_obj_hess, max_obj_hv, max_eval,
                   max_iter, max_time, NaN, Inf, optimality_residual,false,wolfe_step,tau_wolfe,0,true,NaN,
                   false,false,false)
    end
end




function start!(nlp :: AbstractNLPModel,
                s :: TStopping,
                x₀ :: Array{Float64,1} )

    #à priori x0 vit dans l'espace actif
        ∇f₀=grad(nlp,x₀)

    s.optimality0 = s.optimality_residual(∇f₀)
    s.start_time  = time()
    s.optimal = (s.optimality0 < s.atol) | (s.optimality0 <( s.rtol * s.optimality0))
    s.actfeas = minimum(cons(nlp,x₀)) == 0.0

    GOOD = s.optimal && s.actfeas

    return GOOD, ∇f₀
end

function start!(nlp :: AbstractNLPModel,
                s :: TStopping,
                x₀ :: Array{Float64,1},
                runc :: RUncstrnd )

    ∇f₀ = runc.∇f

    s.optimality0 = s.optimality_residual(∇f₀)
    s.start_time  = time()
    s.optimal = (s.optimality0 < s.atol) | (s.optimality0 <( s.rtol * s.optimality0))
    s.actfeas = minimum(runc.cons) == 0.0

    GOOD = s.optimal && s.actfeas

    return GOOD
end

function stop(nlp :: AbstractNLPModel,
              s :: TStopping,
              x :: Array{Float64,1},
              s1d :: Stopping1D,
              runc :: RUncstrnd,
              stepmax :: Float64
              )

    s.iter += 1

    #counts = nlp.counters
    #calls = [counts.neval_obj,  counts.neval_grad, counts.neval_hess, counts.neval_hprod]
    calls = [neval_obj(nlp), neval_grad(nlp), neval_hess(nlp), neval_hprod(nlp)]

    s.optimality = s.optimality_residual(runc.∇fp)

    s.optimal = (s.optimality < s.atol) | (s.optimality <( s.rtol * s.optimality0))
    #optimal = optimality < s.atol +s.rtol*s.optimality0
    s.unbounded =  s1d.unbounded


    # fine grain limits
    max_obj_f  = calls[1] > s.max_obj_f
    max_obj_g  = calls[2] > s.max_obj_grad
    max_obj_H  = calls[3] > s.max_obj_hess
    max_obj_Hv = calls[4] > s.max_obj_hv

    max_total = sum(calls) > s.max_eval

    # global evaluations diagnostic
    max_calls = (max_obj_f) | (max_obj_g) | (max_obj_H) | (max_obj_Hv) | (max_total)

    elapsed_time = time() - s.start_time

    max_iter = s.iter >= s.max_iter
    max_time = elapsed_time > s.max_time

    # global user limit diagnostic
    s.tired = (max_iter) | (max_calls) | (max_time)

    small_step = runc.step <= eps(Float64)
    subpb_fail =! (runc.solved == 0 && !small_step)
    s.sub_pb_solved = !subpb_fail

    s.actfeas = minimum(runc.cons) == 0.0

    #@show ma.sts.wolfe_step, unc.sunc.wolfe_step
    s.wolfe_step = dot(runc.gxp,runc.d) >= s.tau_wolfe*dot(runc.gx,runc.d)

    # return everything. Most users will use only the first four fields, but return
    # the fine grained information nevertheless.
    return (s.optimal && s.actfeas) || s.unbounded || s.tired || runc.step == stepmax || subpb_fail
end

# end of module
end
