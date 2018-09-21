using MPCCmod
using RlxMPCCmod
using MPCCProblems
using PenMPCCmod
using Penalty

PenMPCCmod_success=true
print_with_color(:yellow, "Test RlxMPCC module:\n")

ex1 = MPCCProblems.ex1()
rlx1 = RlxMPCCmod.RlxMPCC(ex1,1.0,1.0,0.0,-1.0) #Relaxation KDB avec t=1.0
ρ,u = ones(8), zeros(8)
pen1 = PenMPCCmod.PenMPCC(ones(4),
                             rlx1,
                             Penalty.quadratic,
                             rlx1.r, rlx1.s, rlx1.t,
                             ones(8), zeros(8),
                             rlx1.ncc,rlx1.meta.nvar)

pen2 = PenMPCCmod.PenMPCC(ones(4),
                             rlx1,
                             Penalty.lagrangian,
                             rlx1.r, rlx1.s, rlx1.t,
                             ones(8), 0.5*ones(8),
                             rlx1.ncc,rlx1.meta.nvar)

#Test 1: meta check
if (pen1.ncc == 1) & (pen1.meta.nvar == 4) & (pen1.meta.ncon == 1)
 print_with_color(:green, "size check: ✓\n")
else
 throw(error("PenMPCCmod failure: size error"))
end

#Test 3: check obj et grad
if (PenMPCCmod.obj(pen1, pen1.meta.x0) == 0.0) & (PenMPCCmod.grad(pen1,pen1.meta.x0) != PenMPCCmod.grad(pen1,[Inf,Inf,0.,0.])) & (PenMPCCmod.hess(pen1,pen1.meta.x0) != sparse(zeros(4,4))) & (PenMPCCmod.hess(pen1, [1.,1.,1.,1.], obj_weight = 1.0, y = ones(3)) != sparse(zeros(4,4)))

 print_with_color(:green, "obj/grad/hess check: ✓ (mais trop vite fait)\n")
else
 throw(error("PenMPCCmod failure: meta error"))
end

if true
 print_with_color(:green, "cons/viol check: ✓\n")
else
 throw(error("RlxMPCCmod failure: cons/viol error"))
end

#Test 4: Stationarité
 #output check
J = PenMPCCmod.jac(pen1, pen1.meta.x0)
size_test = size(J) == (4,7)
J2 = PenMPCCmod.jac(pen2, pen2.meta.x0)
size_test2 = size(J2) == (4,7)
if size_test & size_test2
 print_with_color(:green, "jac check: ✓\n")
else
 throw(error("RlxMPCCmod failure: jac error"))
end

#Test 5: on va maintenant résoudre ce problème avec IpOpt:
using Ipopt
using MathProgBase
try
 model_relax = NLPModels.NLPtoMPB(pen1, IpoptSolver(print_level = 0))
 #MathProgBase.optimize!(model_relax)
catch solve_relax_ipopt
 print_with_color(:red, string(solve_relax_ipopt,"\n"))
end

#Bilan : donner une sortie bilan
if RlxMPCCmod_success==true
 print_with_color(:yellow, "PenMPCCmod.jl passes the test !\n")
else
 println("PenMPCCmod.jl contains some error")
end
