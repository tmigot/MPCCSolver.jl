#@show wnew, ma.wnew #il y a doublon !
import StoppingPenmod.StoppingPen, StoppingPenmod.spen_start!
import StoppingPenmod.spen_stop!, StoppingPenmod.spen_final!

import Stopping.TStopping

function pen_solve(ma      :: ActifMPCC,
                   xjkl     :: Vector,
                   oa      :: OutputALAS;
                   verbose :: Bool = true)

  #I. Initialize the active-set subproblem
  
  #II. Initialize solve structure
  ############################################################################
  ractif = RActif(xjkl)
  actif_start!(ractif, xjkl, ma.rpen, ma.ncc)
  sts    = TStopping(max_iter = ma.paramset.ite_max_viol, atol = ma.spen.atol)
  sts.wolfe_step = ma.spen.wolfe_step
  ############################################################################

  OK = spen_start!(ma.spen, ma.pen, ma.rpen, xjkl)

 while !OK

  #Relaxation rule:
  ma = relaxation_rule(ma, xjkl, sts, verbose) #après uncmin !

  #III. Minization of the penalized problem in the working subspace
  xjkl, ma, ractif, sts = ma.uncmin(ma, xjkl, ractif, sts, oa)

  verbose && print_with_color(:red, "End - Min: l=$(ractif.iter) |x|=$(norm(xjkl,Inf)) |c(x)|=$(ractif.norm_feas) |L'|=$(sts.optimality)  ρ=$(norm(ma.pen.ρ,Inf)) prec=$(sts.atol) \n")

  OK = spen_stop!(ma.spen, ma.pen, ma.rpen, xjkl)
 end
 
  #IV. Final Rending
  pen_update!(ma.pen, ma.rpen, xjkl,
              ractif.sub_pb_solved,
              fx = ractif.fx,
              gx = ractif.gx,
              step = ractif.step,
              wnew = ractif.wnew)

 spen_final!(ma.spen, ma.rpen)
 ma.spen.wolfe_step = sts.wolfe_step ### dans le spen_final!
 ma.sts = sts #vérifier avant d'enlever

 return xjkl, ma
end

#Relaxation rule:
function relaxation_rule(ma,xjk,sts,verbose)

  l_negative = findfirst(x->x<0, ma.rpen.lambda) != 0

  if (sts.wolfe_step || ma.rpen.step == 0.0) && l_negative
  
   relaxation_rule!(ma, xjk, ma.rpen.lambda, ma.rpen.wnew)

   verbose && print_with_color(:yellow, "Active set: $(ma.wcc) \n")
  end

 return ma
end
