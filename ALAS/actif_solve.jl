function actif_solve(ma     :: ActifMPCC,
                     xjk    :: Vector,
                     oa     :: OutputALAS)

 #I. Initialization
 unc = _init_uncstrnd_solve(ma, xjk)

 #II. Start Result and Stopping
 runc_start!(unc.runc, fx = ma.ractif.fx, 
             gx = ma.ractif.gx,
                        ∇f = grad(ma, xjk, ma.ractif.gx),
             step = ma.ractif.step, d = ma.dj)

 OK = start!(ma.pen, ma.sts, xjk, unc.runc)

 #Major Loop
 while !OK

  #III. Initialize State
  _compute_descent_direction!(ma,xjk,unc)
  stepmax, *, * = pas_max(ma, unc.x, unc.runc.d)

  #IV. Solve sub-problem
  x, unc, ols, good_grad = solve_1d(unc; stepmax = stepmax)

  #V. Update the State if needed
  xjk = _update_state_result!(ma, unc, good_grad) #bizarre le good_grad

  #VI. Stopping
  OK = stop(ma.pen, ma.sts, xjk, unc.sunc, unc.runc, stepmax)

  #VII. Update result
  runc_update!(unc.runc, unc.x, fx   = unc.runc.fxp, 
                                gx   = unc.runc.gxp,
                                                          ∇f   = unc.runc.∇fp,
                                d    = evald(ma, unc.runc.d),
                                iter = unc.runc.iter)

  oa_update!(oa, xjk, ma.pen.ρ, ma.ractif.norm_feas, ma.sts.optimality, unc.runc.d, unc.runc.step, ols, unc.runc.fx, ma.n)
 end

 #Final Rending
 _actifmpcc_update!(ma, unc.runc)
 ma.x0 = redx(ma, xjk)

 return xjk, ma
end

############################################################################
#
#function ***
#
############################################################################
function _actifmpcc_update!(ma :: ActifMPCC, runc :: RUncstrnd)

 ma.sts.wolfe_step = runc.solved == 0 && dot(runc.gx,runc.d) >= ma.paramset.tau_wolfe*dot(ma.ractif.gx,runc.d) #WTF ?

 ma.ractif.sub_pb_solved = ma.sts.sub_pb_solved
 ma.ractif.gx = runc.gx
 ma.ractif.fx = runc.fx

 ma.ractif.step = runc.step
 ma.ractif.iter = ma.sts.iter #why ?
 ma.ractif.step = ma.ractif.iter == 0 ? 0.0 : ma.ractif.step

 ma.dj = runc.d

 return ma
end


############################################################################
#
#function ***
#
############################################################################
function _update_state_result!(ma        :: ActifMPCC,
                               unc       :: UncstrndSolve,
                               good_grad :: Bool)

  xjk = evalx(ma,unc.x)
  unc.runc.gxp = grad(ma, xjk)

  if unc.runc.solved == 0 || unc.runc.solved == 3 #que note le 3 ?

    #V. Update the state
    good_grad || (unc.runc.∇fp = grad(ma, unc.x, unc.runc.gxp))

    s = unc.x - unc.runc.x
    y = unc.runc.∇fp - unc.runc.∇f

    #MAJ des paramètres du calcul de la direction
    ma = ma.direction(ma, unc.x, ma.beta, 
                          unc.runc.∇fp, 
                          unc.runc.∇f, 
                          y, unc.runc.d, unc.runc.step)

  else

    unc.runc.∇fp = grad(ma, xjk, unc.runc.gxp)

  end

 return xjk
end

############################################################################
#
#function ***
#
############################################################################
function _init_uncstrnd_solve(ma :: ActifMPCC, xjk :: Vector)

 runc = RUncstrnd(ma.x0)

 sunc = Stopping1D( tau_wolfe      = ma.paramset.tau_wolfe,
                    tau_armijo     = ma.paramset.tau_armijo,
                    armijo_update  = ma.paramset.armijo_update,
                    wolfe_update   = ma.paramset.wolfe_update,
                    ite_max_wolfe  = ma.paramset.ite_max_wolfe,
                    ite_max_armijo = ma.paramset.ite_max_armijo)

 nlp  = ActifModel(x -> ma.pen.nlp.f(evalx(ma,x)), ma.x0;
                   g = x-> grad(ma,x), g! = x-> grad!(ma,x))

 unc  = UncstrndSolve(nlp, ma.x0, runc, sunc, ma.linesearch)

 return unc
end

############################################################################
#
#function ***
#
############################################################################

function _compute_descent_direction!(ma  :: ActifMPCC,
                                     xjk :: Vector,
                                     unc :: UncstrndSolve)

  unc.runc.∇f  = grad(ma, xjk, unc.runc.gx)
  d     = ma.direction(ma, unc.runc.∇f, xjk, redd(ma,unc.runc.d), ma.beta)

  slope = dot(unc.runc.∇f, d)

  if slope > 0.0  # restart with negative gradient

   d = - unc.runc.∇f
   slope =  dot(unc.runc.∇f,d)

  end

  unc.runc.d = d

 return unc
end

###################################################################################
#
#
#
###################################################################################

importall LSDescentMethods

function working_min_proj(ma     :: ActifMPCC,
                          xjk    :: Vector,
                          oa     :: OutputALAS)

 verbose = true
 n = ma.n
 ρ = ma.pen.ρ

 #Initialization
 #################### new structure 
 ht,gradpen,wnew,step = ma.ractif.fx, ma.ractif.gx, ma.ractif.wnew, ma.ractif.step
 ####################

 x_old = ma.x0

  #calcul la direction + Armijo
  (x,d, f, tmp, iter,
  optimal, tired, status,   ma.counters.neval_obj,
  ma.counters.neval_grad, ma.counters.neval_hess)=Newton(ma, x0=ma.x0,
                                                         stp=ma.sts,
                                     Nwtdirection=NwtdirectionSpectral)
 #NwtdirectionLDLt, NwtdirectionSpectral, CG_HZ

 #modifié par on fait, la MAJ si !tired (et le backtracking dans l'autre fonction)
# if !tired && !ma.sts.actfeas   #backtracking:
#   #Calcul du pas maximum:
#   step, wmax, wnew = pas_max(ma, x_old, d)
#   x = x_old + step*d
#   xjk = evalx(ma, x)
#   ma.x0 = xjk
#   #We hit a new constraint:
#   setw(ma, wmax)
#   ma.sts.wolfe_step = false
#
#  verbose && print_with_color(:yellow, "step: $(step) \n")
# elseif !tired
#   ma.x0 = x
#   xjk = evalx(ma, x)
#   ma.sts.wolfe_step = true
# end
if !tired
   ma.x0 = x
   xjk = evalx(ma, x)
   ma.sts.wolfe_step = true
end

 ht,gradpen = objgrad(ma,xjk)
 subpb_fail = tired

 ma.ractif.step = step
 ma.ractif.wnew = wnew
 ma.ractif.sub_pb_solved = !subpb_fail
 ma.ractif.gx = gradpen
 ma.ractif.fx = ht
 ma.ractif.iter = iter

 ma.dj = d
 return xjk, ma
end
