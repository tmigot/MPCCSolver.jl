pbex1 = ex1()

x0 = 1.5 * ones(2)
stop_1 = MPCCStopping(pbex1, MPCCAtX(x0, zeros(1)), optimality_check = WStat)

mpcc_solve(stop_1, verbose = false)

@show stop_1.current_state.x, status(stop_1)

pbex3 = ex3()

x0 = 1.5 * ones(3)
#stop_3 = MPCCStopping(pbex3, WStat, MPCCAtX(x0, zeros(2)), atol = 1e-3)
stop_3 = MPCCStopping(pbex3, MPCCAtX(x0, zeros(2)), atol = 1e-3, optimality_check = (x,y) -> WStat(x,y, actif = 1e-3))

mpcc_solve(stop_3)

@show stop_3.current_state.x, status(stop_3)
