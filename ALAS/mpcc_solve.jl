###################################################################################
#
# MAIN FUNCTION
#
# TO DO : - c'est pénible d'avoir r,s,t au lieu de t de dim <= 3
#
###################################################################################
function solve(mpccsol :: MPCCSolve)

 xk    = mpccsol.xj
 rmpcc = mpccsol.rmpcc
 smpcc = mpccsol.smpcc

 #Initialization
 rlx = _rlx_init(mpccsol, xk)

 start!(rmpcc, mpccsol.mod, xk)
 relax_start!(rlx.rrelax, rlx.mod, rlx.r, rlx.s, rlx.t, rlx.tb, xk,
              c = rmpcc.feas, fx = rmpcc.fx)
 OK = stop_start!(smpcc, mpccsol.mod, xk, rmpcc)

 or = OutputRelaxation(xk, rmpcc)

 #Major Loop
 while OK

  #solve the sub-problem
  xk, rlx, solved, output = _solve_subproblem(rlx, xk, mpccsol, rmpcc) #pourquoi solved ?

  #update the sub-problem
  _rlx_update!(rlx, mpccsol)

  #stopping
  OK = stop!(smpcc, mpccsol.mod, xk, rmpcc, rlx, solved)

  #output
  UpdateOR(or, xk, solved, rlx.r, rlx.s, rlx.t, rlx.prec, rmpcc, output)

 end
 #End Major Loop


 #Final:
 #update the MPCC with the current iterate:
 mpccsol = set_x(mpccsol, xk) 

 final_update!(rmpcc, mpccsol.mod, xk, rlx)
 rmpcc   = final(smpcc, rmpcc)

 final!(or, mpccsol.mod, rmpcc)
 Print(or, mpccsol.mod.n, mpccsol.paramset.verbose)

 return xk, rmpcc, or
end
###################################################################################
"""
Initialize the solve sub-problem structure
"""
function _rlx_init(mpccsol :: MPCCSolve,
                   xk      :: Vector)

 #Initialize the sub-problem
 (r,s,t) = mpccsol.parammpcc.initrst()
 ρ       = mpccsol.parammpcc.rho_init
 prec    = mpccsol.parammpcc.prec_oracle(r, s, t, mpccsol.parammpcc.precmpcc)

 #Initialize the sub-problem Solve structure:
 rlx = RlxMPCCSolve(mpccsol.mod, r, s, t, prec, ρ, mpccsol.paramset, mpccsol.algoset, xk)
 #rlx.xj = vcat(xk,consG(mpccsol.mod,xk),consH(mpccsol.mod,xk))

 return rlx
end
"""
Update of the solve sub-problem structure
"""
function _rlx_update!(rlx    :: RlxMPCCSolve,
                      mpccsol :: MPCCSolve)

  r,s,t = mpccsol.parammpcc.updaterst(rlx.r, rlx.s, rlx.t)

  tb    = mpccsol.paramset.tb(r, s, t)
  rho   = mpccsol.parammpcc.rho_restart(r, s, t, rlx.rho_init)
  prec  = mpccsol.parammpcc.prec_oracle(r, s, t, mpccsol.parammpcc.precmpcc)

 return update_rlx!(rlx, r, s, t, tb, prec, rho)
end

"""
Solve the relaxed sub-problem
"""
function _solve_subproblem(rlx     :: RlxMPCCSolve,
                          xj       :: Vector,
                          mpccsol  :: MPCCSolve,
                          rmpcc    :: RMPCC)

 return mpccsol.parammpcc.solve_sub_pb(rlx,
                                       rmpcc,
                                       mpccsol.name_relax,
                                       xj)
end

