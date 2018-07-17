###################################################################################
#
# MAIN FUNCTION
#
###################################################################################
function solve(mod::MPCC)
 return solve(MPCCSolve(mod,mod.mp.meta.x0))
end

function solve(mpccsol::MPCCSolve)

 #initialization
 (r,s,t)=mpccsol.paramset.initrst()
 rho=mpccsol.paramset.rho_init
 xk=mpccsol.xj

 smpcc=StoppingMPCC(precmpcc=mpccsol.paramset.precmpcc,
                    paramin=mpccsol.paramset.paramin,
                    prec_oracle=mpccsol.paramset.prec_oracle)
 smpcc,OK,or = start(smpcc,mpccsol.mod,xk,r,s,t)

 #Major Loop
 j=0
 while OK

  xk,solved,rho,output = solve_subproblem(mpccsol,r,s,t,rho)

  #met à jour le MPCC avec le nouveau point:
  mpccsol=addInitialPoint(mpccsol,xk[1:mpccsol.mod.n]) 

  (r,s,t)=mpccsol.paramset.updaterst(r,s,t)

  OK = stop(smpcc,mpccsol.mod,xk,r,s,t,output,solved,or)

  j+=1
 end
 #End Major Loop


 #Traitement final :
 mpccsol.paramset.verbose != 0.0 ? warning_print(smpcc) : nothing

 or=final_message(or,smpcc.solved,smpcc.optimal,smpcc.realisable)

 Print(or,mpccsol.mod.n,mpccsol.paramset.verbose)

 nb_eval=[mpccsol.mod.mp.counters.neval_obj,mpccsol.mod.mp.counters.neval_cons,
          mpccsol.mod.mp.counters.neval_grad,mpccsol.mod.mp.counters.neval_hess,
          mpccsol.mod.G.counters.neval_cons,mpccsol.mod.G.counters.neval_jac,
          mpccsol.mod.H.counters.neval_cons,mpccsol.mod.H.counters.neval_jac]

 # output
 return xk, or, nb_eval
end
###################################################################################


"""
Methode pour résoudre le sous-problème relaxé :
"""
function solve_subproblem(inst::MPCCSolve,
                          r::Float64,s::Float64,t::Float64,
                          rho::Vector)
 return inst.solve_sub_pb(inst.mod,r,s,t,rho,inst.name_relax,inst.paramset,inst.algoset,inst.xj) #renvoie xk,stat
end

"""
Function that prints some warning
"""
function warning_print(sts::StoppingMPCC,param::Bool)

	 sts.realisable || print_with_color(:green,"Infeasible solution: (comp,cons)=($(viol_comp(mod,xk)),$(viol_cons(mod,xk)))\n" )
	 sts.solved || print_with_color(:green,"Subproblem failure. NaN in the solution ? $(true in isnan(xk)). Stationary ? $(realisable && optimal)\n")
	 param || sts.realisable || print_with_color(:green,"Parameters too small\n") 
	 sts.solved && sts.realisable && sts.optimal && print_with_color(:green,"Success\n")
 return
end

function final_message(or::OutputRelaxation,
                       solved::Bool,optimal::Bool,realisable::Bool)

 if solved && optimal && realisable
  or=UpdateFinalOR(or,"Success")
 elseif optimal && realisable
  or=UpdateFinalOR(or,"Success (with sub-pb failure)")
 elseif !realisable
  or=UpdateFinalOR(or,"Infeasible")
 elseif realisable && !optimal
  or=UpdateFinalOR(or,"Feasible, but not optimal")
 else
  or=UpdateFinalOR(or,"autres")
 end

 return or
end
