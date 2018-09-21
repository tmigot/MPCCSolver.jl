using MPCCmod
using RlxMPCCmod
using MPCCProblems

RlxMPCCmod_success=true
print_with_color(:yellow, "Test RlxMPCC module:\n")

ex1 = MPCCProblems.ex1()
rlx1 = RlxMPCCmod.RlxMPCC(ex1,1.0,1.0,0.0,-1.0) #Relaxation KDB avec t=1.0

#Test 1: meta check
if (rlx1.ncc == 1) & (rlx1.meta.nvar == rlx1.n) & (rlx1.meta.ncon == 4)
 print_with_color(:green, "size check: ✓\n")
else
 throw(error("RlxMPCCmod failure: size error"))
end

#Test 3: check obj et grad
#ex1 has a linear objective function, so the gradient is constant and hessian vanishes
if (RlxMPCCmod.obj(rlx1,rlx1.meta.x0) == 0.0) & (RlxMPCCmod.grad(rlx1,rlx1.meta.x0) == RlxMPCCmod.grad(ex1,[Inf,Inf])) & (RlxMPCCmod.hess(rlx1,rlx1.meta.x0) == sparse(zeros(2,2)))
 print_with_color(:green, "obj/grad/hess check: ✓\n")
 print_with_color(:red, "hessian of the Lagrangian to be checked\n")
 Htest = RlxMPCCmod.hess(rlx1, [1.,1.], obj_weight = 1.0, y = ones(4))
else
 throw(error("RlxMPCCmod failure: meta error"))
end

if (RlxMPCCmod.cons(rlx1, [1.,1.]) == [-1.0, 1.0, 1.0, 0.0]) & (norm(RlxMPCCmod.viol(rlx1, [1.,1.]),Inf) == 0.0) & (norm(RlxMPCCmod.viol(rlx1, [0.,1.]),Inf) == 0.0) & (norm(RlxMPCCmod.viol(rlx1, [0.,-1.5]),Inf) == 2.5)
 print_with_color(:green, "cons/viol check: ✓\n")
else
 throw(error("RlxMPCCmod failure: cons/viol error"))
end

#Test 4: Stationarité
Spoint = [0.0, 1.0]
Srlx, yS = [-1.,1.], [0,0,0,0,1,0,1,0,0]

optS = norm(RlxMPCCmod.jac(rlx1, Srlx)*yS+RlxMPCCmod.grad(rlx1, Srlx),Inf)
vioS = norm(RlxMPCCmod.viol(rlx1,Srlx),Inf)

if (optS <= 0.) & (vioS <= 0.)
 print_with_color(:green, "jac check: ✓\n")
else
 throw(error("RlxMPCCmod failure: jac error"))
end

#Test 5: on va maintenant résoudre ce problème avec IpOpt:
using Ipopt
using MathProgBase
try
 model_relax = NLPModels.NLPtoMPB(rlx1, IpoptSolver(print_level = 0))
catch solve_relax_ipopt
 print_with_color(:red, string(solve_relax_ipopt,"\n"))
end

#Bilan : donner une sortie bilan
if RlxMPCCmod_success==true
 print_with_color(:yellow, "RlxMPCCmod.jl passes the test !\n")
else
 println("RlxMPCCmod.jl contains some error")
end
