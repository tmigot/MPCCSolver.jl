##############################################################################
#
#
#
##############################################################################
function pen_solve(ps      :: PenMPCCSolve,
                   oa      :: OutputALAS;
                   verbose :: Bool = true)

 #I. Initialize the active-set subproblem
 xjkl = ps.x0

 #II. Initialize solve structure
 ma = _initialize_solve_actifmpccsolve(ps, xjkl)

 #III. Start Result and Stopping
 actif_start!(ma.ractif, xjkl, ps.rpen, ma.ncc)
 OK = spen_start!(ps.spen, ps.pen, ps.rpen, xjkl)

 while !OK

  #Relaxation of the active constraints
  _relaxation_rule!(ma, xjkl, verbose)

  #Minization of the penalized problem in the working subspace
  xjk, ma = ma.uncmin(ma, xjkl, oa) #ma.x0: itéré en dimension active

  #Backtracking if unfeasible
  xjkl = backtracking!(ma,xjk,verbose)

  #Add new constraints
  MAJ = _active_new_constraint!(ma) #on vient de mettre à jour des contraintes

  #output
  verbose && print_with_color(:red, "Iterate: $xjkl \n")
  verbose && print_with_color(:red, "End - Min: l=$(ma.ractif.iter) |x|=$(norm(xjkl,Inf)) |c(x)|=$(ma.ractif.norm_feas) |L'|=$(ma.sts.optimality)  ρ=$(norm(ma.pen.ρ,Inf)) prec=$(ma.sts.atol) \n")

  #stopping
  OK = spen_stop!(ps.spen, ps.pen, ps.rpen, ma.sts, xjkl) || !MAJ

 end
 
 #IV. Final Rending
  _rpen_update!(ps.pen, ps.rpen, xjkl, ma.ractif)
 spen_final!(ps.spen, ps.rpen)
 ps.x0 = xjkl
 _state_update!(ma, ps)

 return xjkl, ps
end

##############################################################################
#
#
#
##############################################################################
function _state_update!(ma :: ActifMPCC, ps :: PenMPCCSolve)

 ps.w = ma.w
 ps.wn1, ps.wn2, ps.w1, ps.w2, ps.wcomp = ma.wn1,ma.wn2,ma.w1,ma.w2,ma.wcomp
 ps.w3, ps.w4, ps.wcomp, ps.w13c, ps.w24c, ps.wc, ps.wcc = ma.w3, ma.w4, ma.wcomp, ma.w13c, ma.w24c, ma.wc, ma.wcc
 ps.dj = ma.dj
 ps.crho = ma.crho
 ps.beta = ma.beta
 ps.Hess = ma.Hess

 return ps
end

##############################################################################
#
#
#
##############################################################################
function _rpen_update!(pen            :: PenMPCC,
                       rpen           :: RPen,
                       xk             :: Vector,
                       ractif         :: RActif)

 return pen_update!(pen, rpen, xk,
                    ractif.sub_pb_solved,
                    fx = ractif.fx,
                    gx = ractif.gx,
                    step = ractif.step,
                    wnew = ractif.wnew,
                    lambda = ractif.lambda)
end

##############################################################################
#
#
#
##############################################################################
function _active_new_constraint!(ma :: ActifMPCC)

   d = redd(ma,ma.dj)
   x_old = ma.x0 - d * ma.ractif.step
   stepmax, wmax, wnew = pas_max(ma, x_old, d)
   MAJ = false

 if ma.ractif.step == stepmax
  setw(ma, wmax)
  ma.ractif.wnew = wnew
  MAJ = true
 else
  ma.ractif.wnew = zeros(Bool,0,0)
 end

 return MAJ
end

##############################################################################
#
#
#
##############################################################################
function backtracking!(ma :: ActifMPCC, xjkl :: Vector, verbose :: Bool)

  if !ma.sts.actfeas

   #Calcul du pas maximum:
   d = redd(ma,ma.dj)
   x_old = ma.x0 - d * ma.ractif.step
   step, wmax, wnew = pas_max(ma, x_old, d)
   x = x_old + step*d
   xjkl = evalx(ma, x)
   ma.x0 = x

   ma.ractif.step = step
   ma.sts.wolfe_step = false
   verbose && print_with_color(:yellow, "step: $(step) \n")
 
  end

 return xjkl
end

##############################################################################
#
#
#
##############################################################################
function _relaxation_rule!(ma :: ActifMPCC, xjk :: Vector, verbose :: Bool)

  #ma.ractif.lambda, l_negative = lsq_computation_multiplier_bool(ma, xjk)
  l_negative = findfirst(x->x<0, ma.ractif.lambda) != 0

  #faire un test qu'on est pas deux pas consécutifs nuls
  if (ma.sts.wolfe_step || ma.ractif.step == 0.0) && l_negative

   relaxation_rule!(ma, xjk, ma.ractif.lambda, ma.ractif.wnew)

   verbose && print_with_color(:yellow, "Active set: $(ma.wcc) \n")
  end

 return ma
end
