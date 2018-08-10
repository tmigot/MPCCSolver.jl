function rlx_solve!(rlx     :: RlxMPCCSolve;
                    verbose :: Bool = true)

 # S0: Projection of the initial point:
 xjk = _slack_complementarity_projection(rlx)

 # S0: Initialize the parameters
 ps, mult = _init_penmpccsolve(rlx, xjk)

 # S2: Initialize Result and Stopping
 pen_start!(ps.pen, ps.rpen, xjk, lambda = mult)
 rlx.spas, GOOD = pas_start!(rlx.nlp.mod, rlx.spas, xjk, ps.rpen)

 oa = OutputALAS(xjk, ps.dj, rlx.spas.feasibility, 
                 rlx.spas.optimality, ps.pen.ρ, ps.rpen.fx)

 # S3: MAJOR LOOP
 while !GOOD

  #solve sub-problem
  xjk, ps = pen_solve(ps, oa)

  #update the parameters
  rlx.spas, UPDATE = pas_rhoupdate!(rlx.nlp.mod, rlx.spas, xjk)
  _update_penalty!(rlx, ps, xjk, UPDATE, verbose)

  #output
  verbose && rlx.spas.tired && print_with_color(:red, "Max ité. Lagrangien \n")

  #stopping
  rlx.spas, GOOD = pas_stop!(rlx.nlp.mod, rlx.spas, xjk, ps.rpen, ps.spen, minimum(ps.pen.ρ))

 end
 #MAJOR LOOP

 #Final rending:
 _final_pen!(rlx, xjk, ps.rpen, ps.pen.ρ)

 stat = ending_test!(rlx.spas, ps.rpen)

 return xjk, stat, ps.rpen, oa
end

##################################################################################
#
#
#
##################################################################################
function _init_penmpccsolve(rlx,xjk)

 n   = rlx.nlp.mod.n
 ncc = rlx.nlp.mod.ncc

 ρ      = rlx.rho_init

 lambda = _lagrange_comp_init(rlx, ρ, xjk, c = rlx.rrelax.feas_cc)
 u      = _lagrange_init(rlx, ρ, xjk, c = rlx.rrelax.feas)
 mult   = vcat(u[2*ncc+1:2*ncc+2*n], lambda) #a first approximation of multipliers

 # S1: Initialization of PenMPCCSolve
 pen, rpen = _initialize_pen_mpcc(rlx, xjk, ρ, u) #create a new problem and result
 ps        = _initialize_solve_penmpcc(rlx, pen, rpen, xjk)

 return ps, mult
end

##################################################################################
#
#
#
##################################################################################
function _final_pen!(rlx, xjk, rpen, ρ)

 rlx = set_x(rlx, xjk)
 rlx.rho_init = ρ
 rlx.rrelax.fx = rpen.fx

 return rlx
end
