##############################################################################
#
#
#
##############################################################################
function _active_new_constraint!(ma :: ActifMPCC)

   d     = redd(ma,ma.dj)
   x_old = ma.x0 - d * ma.ractif.step

   stepmax, wmax, wnew = pas_max(ma, x_old, d)

 if ma.ractif.step == stepmax

  setw(ma, wmax)

  ma.ractif.wnew = wnew
  MAJ = true

 else

  ma.ractif.wnew = zeros(Bool,0,0)
  MAJ = false

 end

 return MAJ
end

##############################################################################
#
#
#
##############################################################################
function _backtracking!(ma      :: ActifMPCC,
                        xjkl    :: Vector,
                        verbose :: Bool)

  if !ma.sts.actfeas

   #Calcul du pas maximum:
   d = redd(ma,ma.dj)
   x_old = ma.x0 - d * ma.ractif.step
   step, *, * = pas_max(ma, x_old, d)
   x = x_old + step*d
   xjkl = evalx(ma, x)
   ma.x0 = x

   ma.ractif.step    = step
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
function _relaxation_rule!(ma      :: ActifMPCC,
                           xjk     :: Vector,
                           verbose :: Bool)

  #ma.ractif.lambda, l_negative = lsq_computation_multiplier_bool(ma, xjk)
  l_negative = findfirst(x->x<0, ma.ractif.lambda) != 0

  #faire un test qu'on est pas deux pas consécutifs nuls
  if (ma.sts.wolfe_step || ma.ractif.step == 0.0) && l_negative

   relaxation_rule!(ma, xjk, ma.ractif.lambda, ma.ractif.wnew)

   verbose && print_with_color(:yellow, "Active set: $(ma.wcc) \n")
  end

 return ma
end
