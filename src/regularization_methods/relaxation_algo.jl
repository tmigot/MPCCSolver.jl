"""
mpcc_solve: Relaxation method for the MPCC.

`mpcc_solve(:: MPCCStopping; verbose :: Bool = true, kwargs...)`

Note: kwargs are passed to the ParamMPCC structure

See also: *ParamMPCC*, *solveIpopt*
"""
function mpcc_solve(stp :: MPCCStopping; verbose :: Bool = true, kwargs...)

 mod = stp.pb
 parammpcc = ParamMPCC(2*(mod.meta.nvar + mod.meta.ncon + mod.meta.ncc); kwargs...)

 fill_in!(stp, stp.current_state.x)
 stp_rlx = _rlx_init(stp, parammpcc)

 OK = start!(stp)

 while !OK

  stp_rlx = rlx_solve(stp_rlx, stp.current_state.x)

  #stopping
  _reforward!(stp_rlx, stp)
  OK = stop!(stp)

  #additional stop if the parameters are too small
  stp.meta.suboptimal = max(stp_rlx.pb.r, stp_rlx.pb.s, stp_rlx.pb.t) < parammpcc.paramin
  #update the sub-problem
  _rlx_update!(stp_rlx, parammpcc)

  #output
  verbose && printstyled("$(stp.meta.nb_of_stop) Iterate: $(stp.current_state.x), r = $(stp_rlx.pb.r), s = $(stp_rlx.pb.s), t = $(stp_rlx.pb.t) \n", color = :red)
  #verbose && print_with_color(:red, "End - Min: l=$(ma.ractif.iter) |x|=$(norm(xjkl,Inf)) |c(x)|=$(ma.ractif.norm_feas) |L'|=$(ma.sts.optimality)  ρ=$(norm(ma.pen.ρ,Inf)) prec=$(ma.sts.atol) \n")

 end

 return stp
end

"""
Refill the mpccstp.current_state using stp.current_state

typeof(stp_rlx.pb) has to be a RlxMPCC
"""
function _reforward!(stp_rlx :: NLPStopping, stp_mpcc :: MPCCStopping)
 #fill_in!(stp_mpcc, stp_nlp.current_state.x)
 ncon, ncc = stp_mpcc.pb.meta.ncon, stp_mpcc.pb.meta.ncc
 r, s, t, tb = stp_rlx.pb.r, stp_rlx.pb.s, stp_rlx.pb.t, stp_rlx.pb.tb

 state = stp_rlx.current_state
 lG, lH = - state.lambda[ncon+1:ncon+ncc], - state.lambda[ncon+ncc+1:ncon+2*ncc]
 lphi   = state.lambda[ncon+2*ncc+1:ncon+3*ncc]
 dp = dphi(state.cx[ncon+1:ncon+ncc], state.cx[ncon+ncc+1:ncon+2*ncc], r, s, t)
 update!(stp_mpcc.current_state, x   = state.x,
                                 fx  = state.fx,
                                 gx  = state.gx,
                                 Hx  = state.Hx,
                                 mu  = state.mu,
                                 cx  = state.cx[1:ncon],
                                 Jx  = state.Jx[1:ncon,:],
                                 lambda = state.lambda[1:ncon],
                                 cGx = state.cx[ncon+1:ncon+ncc] .- tb,
                                 JGx = state.Jx[ncon+1:ncon+ncc,:],
                                 lambdaG = lG - lphi .* dp[1:ncc],
                                 cHx = state.cx[ncon+ncc+1:ncon+2*ncc] .- tb,
                                 JHx = state.Jx[ncon+ncc+1:ncon+2*ncc,:],
                                 lambdaH = lH - lphi .* dp[ncc+1:2*ncc])
 return stp_mpcc
end

"""
rlx_solve: Solve the regularized problem

`rlx_solve(stp :: NLPStopping, x0 :: AbstractVector)`

"""
function rlx_solve(stp :: NLPStopping, x0 :: AbstractVector)
    reinit!(stp, rstate = true, x = x0)
    solveIpopt(stp)
 return stp
end

"""
Initialize the solve sub-problem structure
"""
function _rlx_init(stp :: MPCCStopping, parammpcc :: ParamMPCC)

 #Initialize the sub-problem
 (r,s,t) = parammpcc.initrst()
 tb      = parammpcc.tb(r, s, t)
 #ρ       = parammpcc.rho_init
 prec    = parammpcc.prec_oracle(r, s, t, parammpcc.precmpcc)

 rlx = RlxMPCC(stp.pb, r, s, t, tb)

 ncon, ncc = stp.pb.meta.ncon, stp.pb.meta.ncc
 state = stp.current_state
 dp = dphi(state.cGx, state.cHx, r, s, t)
 rlx_state = NLPAtX(rlx.meta.x0, vcat(state.lambda, state.lambdaG, state.lambdaH, zeros(ncc)),
                                 fx = state.fx,
                                 gx = state.gx,
                                 Hx = state.Hx,
                                 mu = state.mu,
                                 cx = vcat(state.cx,
                                           state.cGx .+ tb,
                                           state.cHx .+ tb,
                                           phi(state.cGx, state.cHx, r, s, t)),
                                 Jx = vcat(state.Jx,
                                           state.JGx,
                                           state.JHx,
                                           diagm(0 => dp[1:ncc]) * state.JGx
                                         + diagm(0 => dp[ncc+1:2*ncc]) * state.JHx))

 stp_rlx = NLPStopping(rlx, rlx_state, optimality_check = KKT, atol = prec, max_cntrs = stp.meta.max_cntrs)

 return stp_rlx
end

"""
Update of the solve sub-problem structure

update atol in the Stopping.
"""
function _rlx_update!(stp :: NLPStopping, parammpcc :: ParamMPCC)

  rlx = stp.pb

  r,s,t = parammpcc.updaterst(rlx.r, rlx.s, rlx.t)
  tb    = parammpcc.tb(r, s, t)
  prec  = parammpcc.prec_oracle(r, s, t, parammpcc.precmpcc)

 return update_rlx!(rlx, r, s, t, tb)
end
