importall LSDescentMethods

function working_min(ma     :: ActifMPCC,
                     xjk    :: Vector,
                     ractif :: RActif,
                     sts    :: TStopping,
                     oa     :: OutputALAS)

 n=ma.n
 ρ = ma.pen.ρ
 dj = ma.dj
 ht,gradpen,wnew,step = ractif.fx, ractif.gx, ractif.wnew, ractif.step

 l = 0
  ∇f = grad(ma, xjk, gradpen) #one eval can be saved

 sts, OK = start!(ma, sts, xjk, ∇f)

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

  sts.unbounded = outputArmijo==2

    ∇f = grad(ma, xjk, gradpen)

  l += 1
  OK, elapsed_time = stop(ma, sts, l, xjk, ht, ∇f)

  oa_update!(oa, xjk, ρ, ractif.norm_feas, sts.optimality, dj, step, ols, ht, n)

  small_step = step == 0.0
  subpb_fail =! (outputArmijo==0 && !small_step)

  OK = OK || subpb_fail
 end

 ###########################################################################
 #Final rending:
 sts.wolfe_step = dot(gradpen,dj) >= ma.paramset.tau_wolfe*dot(ractif.gx,dj)

 ractif.step = step
 ractif.wnew = wnew
 ractif.sub_pb_solved = !subpb_fail
 ractif.gx = gradpen
 ractif.fx = ht
 ractif.iter = l

 ma.dj = dj
 ###########################################################################

 return xjk,ma,ractif,sts
end

###################################################################################
#
#
#
###################################################################################

importall LSDescentMethods

function working_min_proj(ma     :: ActifMPCC,
                          xjk    :: Vector,
                          ractif :: RActif,
                          sts    :: TStopping,
                          oa     :: OutputALAS)

 verbose = true

 n = ma.n
 ρ = ma.pen.ρ
 ht,gradpen,wnew,step = ractif.fx, ractif.gx, ractif.wnew, ractif.step

 x_old = ma.x0

  (x,d, f, tmp, iter,
  optimal, tired, status,   ma.counters.neval_obj,
  ma.counters.neval_grad, ma.counters.neval_hess)=Newton(ma, x0=ma.x0,
                                                         stp=sts,
                                     Nwtdirection=NwtdirectionSpectral)
 #NwtdirectionLDLt, NwtdirectionSpectral, CG_HZ

 if !tired && !sts.actfeas   #backtracking:
   #Calcul du pas maximum:
   step, wmax, wnew = pas_max(ma, x_old, d)
   x = x_old + step*d
   xjk = evalx(ma, x)
   ma.x0 = xjk
   #We hit a new constraint:
   setw(ma, wmax)
   sts.wolfe_step = false

  verbose && print_with_color(:yellow, "step: $(step) \n")
 elseif !tired
   ma.x0 = x
   xjk = evalx(ma, x)
   sts.wolfe_step = true
 end

 ht,gradpen = objgrad(ma,xjk)
 subpb_fail = tired

 ractif.step = step
 ractif.wnew = wnew
 ractif.sub_pb_solved = !subpb_fail
 ractif.gx = gradpen
 ractif.fx = ht
 ractif.iter = iter

 ma.dj = d
 return xjk,ma,ractif,sts
end
