function working_min(ma     :: ActifMPCC,
                     xjk    :: Vector,
                     oa     :: OutputALAS)

 n = ma.n

 #I. Initialization
 unc = _init_uncstrnd_solve(ma, xjk)
 runc = unc.runc

 #II. Start Result and Stopping
 runc_start!(runc, fx = ma.ractif.fx, gx = ma.ractif.gx,
                        ∇f = grad(ma, xjk, ma.ractif.gx),
             step = ma.ractif.step, d = ma.dj)

 OK = start!(ma, ma.sts, xjk, runc.∇f)

 #Major Loop
 l = 0 #dans result
 subpb_fail = false #dans stopping
 while !OK

  #III. Compute a descent direction
  _compute_descent_direction!(ma,xjk,unc)
  stepmax, wmax, wnew = pas_max(ma, unc.x, runc.d)

  #IV. Solve sub-problem
  x, unc, ols, nbarmijo, nbwolfe, good_grad = solve_1d(unc; stepmax = stepmax)

  #VI. Stopping
  runc.solved = 0
  if nbarmijo >= ma.paramset.ite_max_armijo || true in isnan.(x) || true in isnan(runc.fxp)
   runc.solved = 1
   runc.step   = 0.0
  elseif nbwolfe == ma.paramset.ite_max_wolfe
   runc.solved = 2
  end
  small_step = unc.runc.step <= eps(Float64) ? true : false #on fait un pas trop petit

  #Update the iterate
  if runc.solved == 0 || runc.solved == 3

    #V. Update the state
    runc.gxp = grad(ma, evalx(ma, x))

    good_grad || (runc.∇fp = grad(ma, x, runc.gx))
    s = x - unc.x
    y = runc.∇fp - runc.∇f
    #MAJ des paramètres du calcul de la direction
    ma = ma.direction(ma, x, ma.beta, runc.∇fp, runc.∇f, y, runc.d, runc.step)
    #MAJ du scaling
    #if scaling
    # scale = dot(y,s)/dot(y,y)
    #end
    #if scale <= 0.0
    # scale=1.0
    #end

    unc.x = x
    xjk = evalx(ma,unc.x)
  end

  #VI. Stopping
  ma.sts.unbounded = runc.solved==2
  runc.gxp = grad(ma, xjk)
  runc.∇fp = grad(ma, xjk, runc.gxp)

  l += 1
  OK, elapsed_time = stop(ma, ma.sts, l, xjk, runc.fx, runc.∇fp) #virer les l

  subpb_fail =! (runc.solved==0 && !small_step)

  OK = OK || subpb_fail || runc.step == stepmax

  #VII. Update result

  runc.fx = runc.fxp
  runc.gx = runc.gxp
  runc.∇f = runc.∇fp
  runc.d  = evald(ma, runc.d)

  oa_update!(oa, xjk, ma.pen.ρ, ma.ractif.norm_feas, ma.sts.optimality, runc.d, runc.step, ols, runc.fx, n)
 end

 ###########################################################################
 #Final rending:
 ma.sts.wolfe_step = runc.solved == 0 && dot(runc.gx,runc.d) >= ma.paramset.tau_wolfe*dot(ma.ractif.gx,runc.d)

 ma.ractif.sub_pb_solved = !subpb_fail
 ma.ractif.gx = runc.gx
 ma.ractif.fx = runc.fx

 ma.ractif.step = runc.step
 ma.ractif.iter = l
 ma.ractif.step = ma.ractif.iter == 0 ? 0.0 : ma.ractif.step

 ma.x0 = redx(ma, xjk)

 ma.dj = runc.d
 ###########################################################################

 return xjk, ma
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

   d = - runc.∇f
   slope =  dot(runc.∇f,d)

  end

  unc.runc.d = d

 return unc
end

############################################################################
#
#function ***
#
############################################################################
function working_min2(ma     :: ActifMPCC,
                     xjk    :: Vector,
                     oa     :: OutputALAS)

 n=ma.n
 ρ = ma.pen.ρ
 dj = ma.dj
 ht,gradpen,wnew,step = ma.ractif.fx, ma.ractif.gx, ma.ractif.wnew, ma.ractif.step

 l = 0
  ∇f = grad(ma, xjk, gradpen) #one eval can be saved

 ma.sts, OK = start!(ma, ma.sts, xjk, ∇f)

 subpb_fail = false
 #Boucle 1 : Etape(s) de minimisation dans le sous-espace de travail
 while !OK

  #pourquoi il y a small_step ici et plus bas ?
  xjkl,ma.w,dj,step,wnew,outputArmijo,small_step,ols,gradpen,ht=line_search_solve(ma,
                                                                                 xjk,
                                                                                  dj,
                                                                                step,
                                                                             gradpen,
                                                                                  ht)
  xjk = xjkl

  ma.sts.unbounded = outputArmijo==2

    ∇f = grad(ma, xjk, gradpen)

  l += 1
  OK, elapsed_time = stop(ma, ma.sts, l, xjk, ht, ∇f)

  oa_update!(oa, xjk, ρ, ma.ractif.norm_feas, ma.sts.optimality, dj, step, ols, ht, n)

  small_step = step == 0.0

  subpb_fail =! (outputArmijo==0 && !small_step)

  OK = OK || subpb_fail
 end

 ###########################################################################
 #Final rending:
 ma.sts.wolfe_step = dot(gradpen,dj) >= ma.paramset.tau_wolfe*dot(ma.ractif.gx,dj)

 ma.ractif.step = step
 ma.ractif.wnew = wnew
 ma.ractif.sub_pb_solved = !subpb_fail
 ma.ractif.gx = gradpen
 ma.ractif.fx = ht
 ma.ractif.iter = l

 ma.dj = dj
 ###########################################################################

 return xjk, ma
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
