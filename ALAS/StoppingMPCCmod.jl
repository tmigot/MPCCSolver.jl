module StoppingMPCCmod

import MPCCmod.MPCC, MPCCmod.dual_feasibility, MPCCmod.grad

import RMPCCmod.RMPCC

import RlxMPCCSolvemod.RlxMPCCSolve

type StoppingMPCC

 #paramètres pour la résolution du MPCC
 precmpcc    :: Float64
 paramin     :: Float64 #valeur minimal pour les paramères (r,s,t)
 prec_oracle :: Function

 #variables de l'algos
 param       :: Bool  # parameters are too small
 solved      :: Bool # sub_problem solve

 #variables de sorties
 unbounded   :: Bool
 optimal     :: Bool
 realisable  :: Bool

end

function StoppingMPCC(;precmpcc   :: Float64  = 1e-6,
                      paramin     :: Float64  = 1e-15,
                      prec_oracle :: Function = x->x,
                      param       :: Bool     = true,
                      solved      :: Bool     = true,
                      unbounded   :: Bool     = false,
                      optimal     :: Bool     = false,
                      realisable  :: Bool     = false)

 return StoppingMPCC(precmpcc,paramin,prec_oracle,param,solved,
                     unbounded,optimal,realisable)
end

function stop_start!(smpcc :: StoppingMPCC,
                     mod   :: MPCC,
                     xk    :: Vector,
                     rmpcc :: RMPCC)

 realisable = rmpcc.norm_feas <= smpcc.precmpcc

 optimal = realisable && stationary_check(mod, xk, smpcc.precmpcc)
 OK = !(realisable && optimal)

 smpcc.optimal = optimal
 smpcc.realisable = realisable

 return OK
end

import MPCCmod.viol_cons, MPCCmod.viol_comp

function stop!(smpcc  :: StoppingMPCC,
               mod    :: MPCC,
               xk     :: Vector,
               rmpcc  :: RMPCC,
               rlx   :: RlxMPCCSolve)

  r,s,t = rlx.r, rlx.s, rlx.t
  solved = rlx.spas.sub_pb_solved == 0

  #real = rlx.rrelax.norm_feas

  rmpcc.feas = viol_cons(mod,xk)
  rmpcc.feas_cc = viol_comp(mod,xk)
  rmpcc.norm_feas = norm(vcat(rmpcc.feas,rmpcc.feas_cc),Inf)

  real = rmpcc.norm_feas

  f = rlx.rrelax.fx

  smpcc.solved = true in isnan.(xk) ? false : solved
  smpcc.realisable = real <= smpcc.precmpcc

 if smpcc.solved && smpcc.realisable

  dual_feas = stationary_check(mod, xk[1:mod.n], smpcc.precmpcc)
  smpcc.optimal = !isnan(f) && !(true in isnan.(xk)) && dual_feas

 end

 smpcc.param = (t+r+s) > smpcc.paramin

 OK = smpcc.param && !smpcc.optimal
 
 return OK
end

function final(smpcc :: StoppingMPCC, 
               rmpcc :: RMPCC)

 if smpcc.optimal && smpcc.realisable rmpcc.solved =  0 #success
 elseif smpcc.unbounded rmpcc.solved               = -3 #unbounded
 elseif !smpcc.realisable rmpcc.solved             = -2 #Infeasible
 elseif !smpcc.optimal rmpcc.solved                =  1 #Feasible but not optimal
 end

 return rmpcc
end



###################################################################################
#
# Function to verify the dual feasibility
#
# stationary_check(mod::MPCC,x::Vector,precmpcc::Float64)
#
###################################################################################
#Tangi18: est-ce que ça devrait être au MPCCmod de faire ça ?
include("stopping_mpcc_fcts.jl")


#end of module
end
