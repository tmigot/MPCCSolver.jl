"""
+solve(mod::MPCCmod.MPCC)
+solve(MPCCsolveinst::MPCCSolvemod.MPCCSolve)

-solve_subproblem(inst::MPCCSolvemod.MPCCSolve,
                 r::Float64,s::Float64,t::Float64,
                 rho::Vector)
-warning_print(realisable::Bool,solved::Bool,param::Bool,optimal::Bool)
-final_message(or::OutputRelaxationmod.OutputRelaxation,
               solved::Bool,optimal::Bool,realisable::Bool)
"""

module MPCCsolve

using MPCCmod
using OutputRelaxationmod
using MPCCSolvemod
using StoppingMPCCmod

export solve


#TO DO
# - est-ce qu'on pourrait avoir la possibilité d'avoir (r,s,t) en variable ?

###################################################################################
#
# MAIN FUNCTION
#
###################################################################################
function solve(mod::MPCCmod.MPCC)
 return solve(MPCCSolvemod.MPCCSolve(mod,mod.mp.meta.x0))
end

function solve(mpccsol::MPCCSolvemod.MPCCSolve)

 #initialization
 (r,s,t)=mpccsol.paramset.initrst()
 rho=mpccsol.paramset.rho_init
 xk=mpccsol.xj

 sts,OK,or = StoppingMPCCmod.start(mpccsol,xk,r,s,t)

 #Major Loop
 j=0
 while OK

  xk,solved,rho,output = solve_subproblem(mpccsol,r,s,t,rho)

  #met à jour le MPCC avec le nouveau point:
  mpccsol=MPCCSolvemod.addInitialPoint(mpccsol,xk[1:mpccsol.mod.n]) 

  (r,s,t)=mpccsol.paramset.updaterst(r,s,t)

  OK = StoppingMPCCmod.stop(mpccsol,xk,r,s,t,output,solved,or,sts)

  j+=1
 end
 #End Major Loop


 #Traitement final :
 mpccsol.paramset.verbose != 0.0 ? warning_print(sts) : nothing

 or=final_message(or,sts.solved,sts.optimal,sts.realisable)

 OutputRelaxationmod.Print(or,mpccsol.mod.n,mpccsol.paramset.verbose)

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
function solve_subproblem(inst::MPCCSolvemod.MPCCSolve,
                          r::Float64,s::Float64,t::Float64,
                          rho::Vector)
 return inst.solve_sub_pb(inst.mod,r,s,t,rho,inst.name_relax,inst.paramset,inst.algoset,inst.xj) #renvoie xk,stat
end

"""
Function that prints some warning
"""
function warning_print(sts::StoppingMPCCmod.StoppingMPCC,param::Bool)

	 sts.realisable || print_with_color(:green,"Infeasible solution: (comp,cons)=($(MPCCmod.viol_comp(mod,xk)),$(MPCCmod.viol_cons(mod,xk)))\n" )
	 sts.solved || print_with_color(:green,"Subproblem failure. NaN in the solution ? $(true in isnan(xk)). Stationary ? $(realisable && optimal)\n")
	 param || sts.realisable || print_with_color(:green,"Parameters too small\n") 
	 sts.solved && sts.realisable && sts.optimal && print_with_color(:green,"Success\n")
 return
end

function final_message(or::OutputRelaxationmod.OutputRelaxation,
                       solved::Bool,optimal::Bool,realisable::Bool)

 if solved && optimal && realisable
  or=OutputRelaxationmod.UpdateFinalOR(or,"Success")
 elseif optimal && realisable
  or=OutputRelaxationmod.UpdateFinalOR(or,"Success (with sub-pb failure)")
 elseif !realisable
  or=OutputRelaxationmod.UpdateFinalOR(or,"Infeasible")
 elseif realisable && !optimal
  or=OutputRelaxationmod.UpdateFinalOR(or,"Feasible, but not optimal")
 else
  or=OutputRelaxationmod.UpdateFinalOR(or,"autres")
 end

 return or
end

#end of module
end
