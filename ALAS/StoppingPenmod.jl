module StoppingPenmod

import RPenmod.RPen

importall PenMPCCmod
import PenMPCCmod.PenMPCC

import Stopping.TStopping

type StoppingPen

    atol :: Float64                  # absolute tolerance
    rtol :: Float64                  # relative tolerance
    goal_viol :: Float64   # below this value, the problem is declared unbounded
    rho_max :: Float64
    # fine grain control on ressources
    max_obj_f :: Int                 # max objective function (f) evaluations allowed
    max_obj_grad :: Int              # max objective gradient (g) evaluations allowed
    max_obj_hess :: Int              # max objective hessian (H) evaluations allowed
    max_obj_hv :: Int                # max objective H*v (HV) evaluations allowed
    # global control on ressources
    max_eval :: Int                  # max evaluations (f+g+H+Hv) allowed
    max_iter :: Int                  # max iterations allowed
    max_time :: Float64              # max elapsed time allowed
    # global information to the stopping manager
    start_time :: Float64            # starting time of the execution of the method
    #dual Lagrangian
    optimality :: Float64           # value of the optimality residual at starting point
    optimality_residual :: Function  # function to compute the optimality residual
    optimal :: Bool
    #Sign lambda
    l_negative :: Bool
    negative_test :: Function #findfirst(x->x<0,[lg;lh;lphi])!=0
    wolfe_step :: Bool
    #Feasibility
    feasibility :: Float64
    feasibility_residual :: Function #norm(MPCCmod.viol_contrainte(alas.mod,xjk),Inf)
    feas :: Bool
    # Stopping properties
    tired::Bool

    function StoppingPen(;atol :: Float64 = 1.0e-8,
                       rtol :: Float64 = 1.0e-6,
                       goal_viol :: Float64 = -1.0e50,
                       rho_max :: Float64 = 1.0e20,
                       max_obj_f :: Int = typemax(Int),
                       max_obj_grad :: Int = typemax(Int),
                       max_obj_hess :: Int = typemax(Int),
                       max_obj_hv :: Int = typemax(Int),
                       max_eval :: Int = 100000,
                       max_iter :: Int = 5000,
                       max_time :: Float64 = 600.0, # 10 minutes
                       optimality_residual :: Function = x -> norm(x,Inf),
                       negative_test :: Function = l -> findfirst(x->x<0,l)!=0,
                       feasibility_residual :: Function = x -> norm(x,Inf),
                       tired::Bool=false,
                       kwargs...)
        
        return new(atol, rtol, goal_viol,rho_max,
                   max_obj_f, max_obj_grad, max_obj_hess, max_obj_hv, max_eval,
                   max_iter, max_time, NaN, Inf, optimality_residual,false,false,
                   negative_test,false,NaN,feasibility_residual,false,tired)
    end
end

function spen_start!(spen :: StoppingPen,
                    pen  :: PenMPCC,
                    rpen :: RPen,
                    xk   :: Vector)

 OK = false

 return OK
end

function spen_stop!(spen :: StoppingPen, 
                   pen  :: PenMPCC,
                   rpen :: RPen,
                   xk   :: Vector)

 OK = true
 
 return OK
end

function spen_final!(spen :: StoppingPen, 
                     rpen :: RPen)

 #spen.wolfe_step = sts.wolfe_step

 return spen
end

# end of module
end
