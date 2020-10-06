test1 = ex1()
#nlp = convert(AbstractNLPModel, test1)
nlp = NLMPCC(test1)
x0 = [1.5, 1.5]

@test nlp.mod.meta.ncc  == 1
@test nlp.mod.meta.ncon == 1
@test obj(nlp, nlp.meta.x0) == 0.0
@test grad(nlp, nlp.meta.x0) == grad(test1,[Inf,Inf])
@test hess(nlp, nlp.meta.x0) == sparse(zeros(2,2))

nlp_at_x = NLPAtX(x0, zeros(4))
stop_2 = NLPStopping(nlp, KKT, nlp_at_x)

printstyled("1st scenario:\n")
solveIpopt(stop_2)
@show stop_2.current_state.x, status(stop_2)

stop_s = MPCCStopping(test1, SStat, MPCCAtX(x0, zeros(1)))

printstyled("2nd scenario:\n")
solveIpopt(stop_s)
@show stop_s.current_state.x, status(stop_s)

##############################################################################

test2 = ex3()
#nlp = convert(AbstractNLPModel, test1)
nlp = NLMPCC(test2)
x0 = [1.5, 1.5, 1.5]

@test nlp.mod.meta.ncc  == 1
@test nlp.mod.meta.ncon == 2

nlp_at_x = NLPAtX(x0, zeros(5))
stop_3 = NLPStopping(nlp, KKT, nlp_at_x)

printstyled("1st scenario:\n")
solveIpopt(stop_3)
@show stop_3.current_state.x, status(stop_3)

stop_s = MPCCStopping(test2, MStat, MPCCAtX(x0, zeros(2)))

printstyled("2nd scenario:\n")
solveIpopt(stop_s)
@show stop_s.current_state.x, status(stop_s)
