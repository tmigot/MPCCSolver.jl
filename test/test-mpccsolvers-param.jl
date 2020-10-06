nbc = 3
p = ParamMPCC(nbc)

@test p.precmpcc == 1e-3
#@test p.prec_oracle == (r,s,t,prec) -> max(min(r,s,t),prec)
#@test p.rho_restart == (r,s,t,rho) -> rho
@test p.paramin == sqrt(eps(Float64))
@test p.initrst() == (1.0,0.5,0.5)
@test p.updaterst(1.0,1.0,1.0) == (0.1,0.1,0.05)
#@test p.solve_sub_pb == x -> x
@test p.tb(1.0,1.0,1.0) == 1.0
@test p.ite_max_alas == 100
@test p.ite_max_viol == 30
@test p.rho_init == ones(nbc)
@test p.rho_update == 1.5
@test p.rho_max == 4000.0
@test p.goal_viol == 0.75
@test p.ite_max_armijo == 100
@test p.tau_armijo == 0.1
@test p.armijo_update == 0.9
@test p.ite_max_wolfe == 10
@test p.tau_wolfe == 0.6
@test p.wolfe_update == 5.0
@test p.verbose == 0
#@test p.uncmin == x -> x
#@test p.penalty == x -> x
#@test p.direction == x -> x
#@test p.linesearch == x -> x
