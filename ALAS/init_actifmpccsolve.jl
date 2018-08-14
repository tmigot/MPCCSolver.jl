function _initialize_solve_actifmpccsolve(ps      :: PenMPCCSolve,
                                          xjkl    :: Vector)

 ractif = RActif(xjkl)

 sts    = TStopping(max_iter   = ps.paramset.ite_max_viol,
                    atol       = ps.spen.atol,
                    wolfe_step = ps.spen.wolfe_step)

 ma = ActifMPCC(ps.pen,
                xjkl,
                ps.ncc,
                ps.paramset,
                ps.algoset.uncmin,
                ps.algoset.direction,
                ps.algoset.linesearch,
                sts,
                ractif)

  #Initialize the state
  setw(ma, ps.w) #met à jour ma.wnew !?
  ma.wnew = ps.rpen.wnew #zeros(Bool,0,0)
  ma.ractif.wnew = ma.wnew

  ma.dj = ps.dj
  ma.crho = ps.crho
  ma.beta = ps.beta
  ma.Hess = ps.Hess

 #Create an ActifMPCC
 return ma
end
