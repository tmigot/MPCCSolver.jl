rlxex1 = ex1()
rlx1 = MPCCSolver.RlxMPCC(rlxex1, 1.0, 1.0, 0.0, 1.0) #Relaxation KDB avec t=1.0

#Test 1: meta check
@test rlx1.mod.meta.ncc == 1
@test rlx1.meta.nvar == 2
@test rlx1.meta.ncon == 4

#Test 3: check obj et grad
#rlxex1 has a linear objective function, so the gradient is constant and hessian vanishes
@test MPCCSolver.obj(rlx1,rlx1.meta.x0) == 0.0
@test MPCCSolver.grad(rlx1,rlx1.meta.x0) == MPCCSolver.grad(rlxex1,[Inf,Inf])
@test MPCCSolver.hess(rlx1,rlx1.meta.x0) == sparse(zeros(2,2))
#Htest = MPCCSolver.hess(rlx1, [1.,1.], obj_weight = 1.0, y = ones(4))

@test MPCCSolver.cons(rlx1, [1.,1.])             == [-1.0, 2.0, 2.0, 0.0]
@test norm(MPCCSolver.viol(rlx1, [1.,1.]),Inf)   == 0.0
@test norm(MPCCSolver.viol(rlx1, [0.,1.]),Inf)   == 0.0
@test norm(MPCCSolver.viol(rlx1, [0.,-1.5]),Inf) == 2.5

#Test 4: Stationarité
Spoint = [0.0, 1.0]
Srlx, yS = [-1.,1.], [-1.,-1.,0.,0.]

@test norm(MPCCSolver.jac(rlx1, Srlx)'*yS+MPCCSolver.grad(rlx1, Srlx),Inf) == 0.
@test norm(MPCCSolver.viol(rlx1,Srlx),Inf) == 0.0

#Test 5: on va maintenant résoudre ce problème avec IpOpt:
x0 = [1.0, 1.0]
stats = ipopt(rlx1, print_level = 0, x0 = x0)
#Use y0 (general), zL (lower bound), zU (upper bound)
#for initial guess of Lagrange multipliers.
@show stats.solution, stats.status

nlp_at_x = NLPAtX(x0)
stop = NLPStopping(rlx1, KKT, nlp_at_x)

solveIpopt(stop)
@show stop.current_state.x, status(stop)
