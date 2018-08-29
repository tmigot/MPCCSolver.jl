function _initialize_solve_penmpcc(rlx  :: RlxMPCCSolve,
                                   penmpcc :: PenMPCC,
                                   rpen    :: RPen,
                                   xj      :: Vector)

 spen   = StoppingPen(max_iter = rlx.paramset.ite_max_viol, atol = rlx.prec)

 ps = PenMPCCSolve(penmpcc,
                   xj,
                   penmpcc.ncc,
                   rlx.paramset,
                   rlx.algoset,
                   spen,
                   rpen)

 #Create an ActifMPCC
 return ps
end
