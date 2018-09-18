using MPCCmod
using MPCCProblems

MPCCmod_success=true
print_with_color(:yellow, "Test MPCC module:\n")

ex1bis=JuMP.Model()
ux(i)=[100;100][i]
JuMP.@variable(ex1bis,x[i=1:2], upperbound=ux(i),start=1.0)
JuMP.@NLobjective(ex1bis,Min,x[1]-x[2])
JuMP.@constraint(ex1bis,1-x[2]>=0)
ex1bis = MathProgNLPModel(ex1bis)

ex1 = MPCCProblems.ex1()
ex1bis = MPCCmod.MPCC(ex1bis)

#Test 1: meta check
if (ex1.meta.ncc == 1) & (ex1bis.meta.ncc == 0)  & (ex1.meta.nvar == 2) & (ex1bis.meta.nvar == 2)
 print_with_color(:green, "size check: ✓\n")
else
 throw(error("MPCCmod failure: size error"))
end

#Test 2: data check
#ex1 has no bounds constraints, one >= constraint, and complementarity constraints
if (ex1.meta.lvar == -Inf*ones(2)) & (ex1.meta.uvar == Inf*ones(2)) & (ex1.meta.ucon == Inf*ones(1)) & (ex1.meta.lccG == zeros(1)) & (ex1.meta.lccH == zeros(1))
 print_with_color(:green, "meta check: ✓\n")
else
 throw(error("MPCCmod failure: meta error"))
end

#Test 3: check obj et grad
#ex1 has a linear objective function, so the gradient is constant and hessian vanishes
if (MPCCmod.obj(ex1,ex1.meta.x0) == 0.0) & (MPCCmod.grad(ex1,ex1.meta.x0) == MPCCmod.grad(ex1,[Inf,Inf])) & (MPCCmod.hess(ex1,ex1.meta.x0) == sparse(zeros(2,2)))
 print_with_color(:green, "obj/grad/hess check: ✓\n")
else
 throw(error("MPCCmod failure: meta error"))
end

#Test 4: check les contraintes et la réalisabilité
#ex1 has two stationary points:
Wpoint, Spoint = zeros(2), [0.0, 1.0]
if (norm(MPCCmod.viol(ex1,Spoint),Inf) == 0.0) & (norm(MPCCmod.viol(ex1,Wpoint),Inf) == 0.0) & (MPCCmod.cons(ex1,Wpoint) == zeros(4)) & (norm(MPCCmod.viol_comp(ex1,Wpoint),Inf)==0.0)
 print_with_color(:green, "cons/viol check: ✓\n")
else
 throw(error("MPCCmod failure: cons/viol error"))
end

#Test 5: check the jacobian
# Wpoint is an W-stationary point, and Spoint is an S-stationary point
lambdaW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]
AW = MPCCmod.jac(ex1, Wpoint)
aw = MPCCmod.grad(ex1, Wpoint)
lambdaS = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
AS = MPCCmod.jac(ex1, Spoint)
bs = MPCCmod.grad(ex1, Spoint)

if (norm(aw + AW*lambdaW,Inf) == 0.0) & (norm(bs + AS*lambdaS,Inf) == 0.0) & (norm(AW\(-aw)-lambdaW,Inf) == 0.0) & (norm(AS\(-bs)-lambdaS,Inf) == 0.0)
 print_with_color(:green, "stationarity check: ✓\n")
else
 throw(error("MPCCmod failure: jac error"))
end

#Bilan : donner une sortie bilan
if MPCCmod_success==true
 print_with_color(:yellow, "MPCCmod.jl passes the test !\n")
else
 println("MPCCmod.jl contains some error")
end
