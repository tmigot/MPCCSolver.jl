##########################################################################

#Using Stopping, the idea is to create a buffer function
"""
solveIpopt: Use Ipopt to solve the problem. If stp is an MPCCStoping the MPCC
is set as a NLP using NLMPCC.

`solveIpopt(:: Union{NLPStopping,MPCCStopping}; print_level :: Int = 0)`
"""
function solveIpopt(stp :: Union{NLPStopping,MPCCStopping}; print_level :: Int = 0)

 if typeof(stp.pb) <: AbstractMPCCModel
     nlp = NLMPCC(stp.pb)
 else #typeof(stp.pb) == NLPModels
     nlp = stp.pb
 end
 #xk = solveIpopt(stop.pb, stop.current_state.x)
 stats = ipopt(nlp, print_level     = print_level,
                    tol             = stp.meta.rtol,
                    x0              = stp.current_state.x, #temporary fix
                    max_iter        = stp.meta.max_iter,
                    max_cpu_time    = stp.meta.max_time,
                    dual_inf_tol    = stp.meta.atol,
                    constr_viol_tol = stp.meta.atol,
                    compl_inf_tol   = stp.meta.atol)

 stp.meta.nb_of_stop = stats.iter
 #stats.elapsed_time

 x = stats.solution

 #Not mandatory, but in case some entries of the State are used to stop
  fill_in!(stp, x)
  stop!(stp)

  #Update the meta boolean with the output message
  if stats.status == :first_order stp.meta.suboptimal      = true end
  if stats.status == :acceptable  stp.meta.suboptimal      = true end
  if stats.status == :infeasible  stp.meta.infeasible      = true end
  if stats.status == :small_step  stp.meta.stalled         = true end
  if stats.status == :max_iter    stp.meta.iteration_limit = true end
  if stats.status == :max_time    stp.meta.tired           = true end

 return stp
end
