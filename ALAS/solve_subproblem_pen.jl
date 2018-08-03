#@show wnew, ma.wnew #il y a doublon !
function solve_subproblem_pen(ma      :: ActifMPCC,
                              xjk     :: Vector,
                              oa      :: OutputALAS;
                              verbose :: Bool = true)

  #I. Initialize the active-set subproblem
  #Relaxation rule:
  ma = relaxation_rule(ma, xjk, verbose)
  
  #II. Initialize result (actif_start!)
  ractif = RActif(xjk)

  actif_start!(ractif, xjk, ma.rpen, ma.ncc)

  #III. Minization of the penalized problem in the working subspace
  xjkl, ma, ractif = working_min(ma, xjk, ractif, oa)

  verbose && print_with_color(:red, "End - Min: l=$(ractif.iter) |x|=$(norm(xjkl,Inf)) |c(x)|=$(ractif.norm_feas) |L'|=$(ma.sts.optimality)  ρ=$(norm(ma.pen.ρ,Inf)) prec=$(ma.sts.atol) \n")
 
  #IV. Final Rending
  pen_update!(ma.pen, ma.rpen, xjkl,
              ractif.sub_pb_solved,
              fx = ractif.fx,
              gx = ractif.gx,
              step = ractif.step,
              wnew = ractif.wnew)

 return xjkl, ma
end


function relaxation_rule(ma,xjk,verbose)

  #Relaxation rule:
  l_negative = findfirst(x->x<0, ma.rpen.lambda) != 0
  if (ma.sts.wolfe_step || ma.rpen.step == 0.0) && l_negative
  
   relaxation_rule!(ma, xjk, ma.rpen.lambda, ma.rpen.wnew)

   verbose && print_with_color(:yellow, "Active set: $(ma.wcc) \n")
  end

 return ma
end
