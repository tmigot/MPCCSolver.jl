function rlx_solve!(rlx     :: RlxMPCCSolve;
                    verbose :: Bool = true)

 n   = rlx.mod.n
 ncc = rlx.mod.ncc

 # S0: Projection of the initial point:
 xjk = _slack_complementarity_projection(rlx)

 # S0: Initialize the parameters
 ρ      = rlx.rho_init

 lambda = _lagrange_comp_init(rlx, ρ, xjk, c = rlx.rrelax.feas_cc)
 u      = _lagrange_init(rlx, ρ, xjk, c = rlx.rrelax.feas)
 mult   = vcat(u[2*ncc+1:2*ncc+2*n], lambda)

 # S1: Initialization of ActifMPCC 
 pen, rpen = _initialize_pen_mpcc(rlx, xjk, ρ, u) #create a new problem and result
 ma        = _initialize_solve_penmpcc(rlx, pen, rpen, xjk)

 # S2: Initialize Result and Stopping
 pen_start!(ma.pen, ma.rpen, xjk, lambda = mult)
 rlx.spas, GOOD = pas_start!(rlx.mod, rlx.spas, xjk, ma.rpen)

 oa = OutputALAS(xjk, ma.dj, rlx.spas.feasibility, 
                 rlx.spas.optimality,ma.pen.ρ, ma.rpen.fx)

 # S3: MAJOR LOOP
 while !GOOD

  xjk, ma = pen_solve(ma, xjk, oa) #xjk dans ma.x0 ?

  rlx.spas, UPDATE = pas_rhoupdate!(rlx.mod, rlx.spas, xjk)

  _update_penalty!(rlx, ma, xjk, UPDATE, verbose) #modifie ma et rlx

  verbose && rlx.spas.tired && print_with_color(:red, "Max ité. Lagrangien \n")

  rlx.spas, GOOD = pas_stop!(rlx.mod, rlx.spas, xjk, ma.rpen, ma.sts, minimum(ma.pen.ρ))

 end
 #MAJOR LOOP

 #Traitement finale :
 rlx = set_x(rlx, xjk)

 stat = ending_test!(rlx.spas, ma.rpen, ma.sts)

 return xjk, stat, ma.rpen, oa
end
#ma.sts:
