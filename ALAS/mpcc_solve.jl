###################################################################################
#
# MAIN FUNCTION
#
# TO DO : c'est pénible d'avoir r,s,t au lieu de t de dim <= 3
#
###################################################################################
function solve(mpccsol :: MPCCSolve)

 #Initialization
 (r,s,t) = mpccsol.parammpcc.initrst()
 rho = mpccsol.parammpcc.rho_init
 xk = mpccsol.xj

 rmpcc = RMPCC(xk)
 smpcc = StoppingMPCC(precmpcc = mpccsol.parammpcc.precmpcc,
                    paramin = mpccsol.parammpcc.paramin,
                    prec_oracle = mpccsol.parammpcc.prec_oracle)

 start!(rmpcc, mpccsol.mod, xk)
 OK = stop_start!(smpcc, mpccsol.mod, xk, rmpcc, r, s, t)

 or = OutputRelaxation(xk, rmpcc)

 #Major Loop
 j = 0
 while OK

  rho = j == 0 ? mpccsol.parammpcc.rho_restart(r,s,t,rho) : rho

  xk,solved,rho,output = solve_subproblem(mpccsol,r,s,t,rho)

  update!(rmpcc, mpccsol.mod, xk[1:mpccsol.mod.n])

  #met à jour le MPCC avec le nouveau point:
  mpccsol = set_x(mpccsol, xk[1:mpccsol.mod.n]) 

  (r,s,t) = mpccsol.parammpcc.updaterst(r,s,t)

  OK = stop!(smpcc,mpccsol.mod,xk,rmpcc,r,s,t,solved)

  UpdateOR(or,xk[1:mpccsol.mod.n],0,r,s,t,
              smpcc.prec_oracle(r,s,t,smpcc.precmpcc),
              rmpcc.norm_feas,output,rmpcc.fx)

  j+=1
 end
 #End Major Loop


 #Traitement final :
 rmpcc = final(smpcc,rmpcc)
 or = final!(or,mpccsol.mod,rmpcc)

 Print(or,mpccsol.mod.n,mpccsol.paramset.verbose)

 # output
 return xk, rmpcc, or
end
###################################################################################

"""
Une méthode supplémentaire rapide
"""
function solve(mod :: MPCC)

 return solve(MPCCSolve(mod, mod.mp.meta.x0))
end


"""
Methode pour résoudre le sous-problème relaxé :
"""
function solve_subproblem(inst :: MPCCSolve,
                          r    :: Float64,
                          s    :: Float64,
                          t    :: Float64,
                          rho  :: Vector)


 #Il faut réflechir un peu plus sur des alternatives
 prec = inst.parammpcc.prec_oracle(r,s,t,inst.parammpcc.precmpcc)

 return inst.parammpcc.solve_sub_pb(inst.mod,
                                    r,s,t,
                                    rho,
                                    inst.name_relax,
                                    inst.paramset,
                                    inst.algoset,
                                    inst.xj,
                                    prec)
end

"""
****
"""

