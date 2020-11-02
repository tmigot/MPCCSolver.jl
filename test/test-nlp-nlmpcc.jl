x0  = 1.5 * ones(6)
nlp = ADNLPModel(rosenbrock,  x0)

#The traditional way to solve an optimization problem using NLPModelsIpopt
#https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl
printstyled("Oth scenario:\n")

stats = ipopt(nlp, print_level = 0, x0 = x0)
#Use y0 (general), zL (lower bound), zU (upper bound)
#for initial guess of Lagrange multipliers.
@show stats.solution, stats.status

nlp_at_x = NLPAtX(x0)
stop = NLPStopping(nlp, nlp_at_x, optimality_check = unconstrained_check)

#1st scenario, we solve again the problem with the buffer solver
printstyled("1st scenario:\n")
solveIpopt(stop)
@show stop.current_state.x, status(stop)
nbiter = stop.meta.nb_of_stop

#2nd scenario: we check that we control the maximum iterations.
printstyled("2nd scenario:\n")
#rstate is set as true to allow reinit! modifying the State
reinit!(stop, rstate = true, x = x0)
stop.meta.max_iter = max(nbiter-4,1)

solveIpopt(stop)
#Final status is :IterationLimit
@show stop.current_state.x, status(stop)

printstyled("3rd scenario:\n")
mpcc = MPCCNLPs(nlp)
test = NLMPCC(mpcc)
nlp_at_x = NLPAtX(x0)
stop_2 = NLPStopping(test, nlp_at_x, optimality_check = unconstrained_check)

solveIpopt(stop_2)
@show stop_2.current_state.x, status(stop_2)

printstyled("4th scenario:\n")
nlp_at_x = MPCCAtX(mpcc.meta.x0, zeros(0))
stop_nlp = MPCCStopping(mpcc, nlp_at_x, optimality_check = SStat)

@test mpcc.meta.ncc  == 0
@test mpcc.meta.ncon == 0
@test status(stop_nlp) == :Unknown

solveIpopt(stop_nlp)
#Final status is :IterationLimit
@show stop_nlp.current_state.x, status(stop_nlp)
