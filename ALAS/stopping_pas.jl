# A stopping manager for iterative solvers
export TStoppingPAS, start!, stop, ending_test

import RPenmod.RPen
import StoppingPenmod.StoppingPen
import MPCCmod.viol

type TStoppingPAS
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

    #Feasibility
    feasibility :: Float64
    feasibility_residual :: Function #norm(MPCCmod.viol(alas.mod,xjk),Inf)
    feas :: Bool
    # Stopping properties
    tired::Bool
    unbounded           :: Bool

    #bilan
    sub_pb_solved :: Int64

    function TStoppingPAS(;atol :: Float64 = 1.0e-8,
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
                   negative_test,NaN,feasibility_residual,false,tired,false,0)
    end
end




function pas_start!( mod :: MPCCmod.MPCC,
                 s :: TStoppingPAS,
                x₀ :: Array{Float64,1},
                    rpen :: RPen)

    s.l_negative=s.negative_test(rpen.lambda)

    s.feasibility=s.feasibility_residual(rpen.feas)
    s.feas = (s.feasibility < s.atol) | (s.feasibility <( s.rtol * s.feasibility))

    s.optimality = s.optimality_residual(rpen.dual_feas)
    s.optimal = (s.optimality < s.atol) | (s.optimality <( s.rtol * s.optimality))

    s.start_time  = time()

    OK = s.feas && s.optimal && s.l_negative
    return s, OK
end

function pas_rhoupdate!( mod :: MPCCmod.MPCC,
                 s :: TStoppingPAS,
                x₀ :: Array{Float64,1})

    feasibility = s.feasibility
    s.feasibility = s.feasibility_residual(MPCCmod.viol(mod, x₀))
    s.feas = (s.feasibility < s.atol) | (s.feasibility <( s.rtol * s.feasibility))

    UPDATE = s.feasibility>s.goal_viol*feasibility && !s.feas

    return s, UPDATE
end

function pas_stop!(mod :: MPCCmod.MPCC,
                   s :: TStoppingPAS,
                   x  :: Array{Float64,1},
                   rpen :: RPen,
                   spen :: StoppingPen,
                   ρ :: Float64)

    iter = rpen.iter
    s.unbounded = spen.unbounded

    #counts = nlp.counters
    #calls = [counts.neval_obj,  counts.neval_grad, counts.neval_hess, counts.neval_hprod]
    calls = [neval_obj(mod.mp), neval_grad(mod.mp), neval_hess(mod.mp), neval_hprod(mod.mp)]

    s.feasibility=s.feasibility_residual(MPCCmod.viol(mod,x))
    s.feas = (s.feasibility < s.atol) | (s.feasibility <( s.rtol * s.feasibility))

    s.optimality = s.optimality_residual(rpen.dual_feas)
    s.optimal = (s.optimality < s.atol) | (s.optimality <( s.rtol * s.optimality))


    # fine grain limits
    max_obj_f  = calls[1] > s.max_obj_f
    max_obj_g  = calls[2] > s.max_obj_grad
    max_obj_H  = calls[3] > s.max_obj_hess
    max_obj_Hv = calls[4] > s.max_obj_hv

    max_total = sum(calls) > s.max_eval

    # global evaluations diagnostic
    max_calls = (max_obj_f) | (max_obj_g) | (max_obj_H) | (max_obj_Hv) | (max_total)

    elapsed_time = time() - s.start_time

    max_iter = iter >= s.max_iter

    max_time = elapsed_time > s.max_time

    # global user limit diagnostic
    s.tired = (max_iter) | (max_calls) | (max_time)

    #si on bloque mais qu'un multiplicateur est <0 on continue
    subpb_fail=!rpen.sub_pb_solved && !s.l_negative #devrait être spen

    @show s.tired, subpb_fail, s.l_negative, s.feas, s.optimal, s.unbounded
    OK = s.tired || subpb_fail || (!s.l_negative && (s.feas || ρ == s.rho_max ) && s.optimal) || s.unbounded

    # return everything. Most users will use only the first four fields, but return
    # the fine grained information nevertheless.
    return s, OK
           #,max_obj_f, max_obj_g, max_obj_H, max_obj_Hv, max_total, max_iter, max_time
end

function ending_test!(spas::TStoppingPAS,
                     rpen::RPen)

 sub_pb, unbounded = !rpen.sub_pb_solved, spas.unbounded #devrait être spen

 stat=0
 if unbounded
  print_with_color(:red, "Unbounded Subproblem\n")
  stat=2
 else
  if sub_pb
   print_with_color(:red, "Failure : Armijo Failure\n")
   stat=-1
  end
  if !spas.feas #feas>alas.prec
   print_with_color(:red, "Failure : Infeasible Solution. norm: $(spas.feas)\n")
   stat=1
  end
  if !spas.optimal #dual_feas>alas.prec
   if spas.tired #k>=alas.paramset.ite_max_alas
    print_with_color(:red, "Failure : Non-optimal Sol. norm: $(spas.optimality)\n")
    stat=-2
   else
    print_with_color(:red, "Inexact : Fritz-John Sol. norm: $(spas.optimality)\n")
    stat=1
   end
  end
 end

 spas.sub_pb_solved = stat
 return stat
end
