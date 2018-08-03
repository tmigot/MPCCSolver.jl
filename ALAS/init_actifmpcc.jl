function _initialize_solve_penmpcc(rlx  :: RlxMPCCSolve,
                                   penmpcc :: PenMPCC,
                                   rpen    :: RPen,
                                   xj      :: Vector)

 sts    = TStopping(max_iter = rlx.paramset.ite_max_viol, atol = rlx.prec)

 #Create an ActifMPCC
 return ActifMPCC(penmpcc,
                  penmpcc.ncc,
                  rlx.paramset,
                  rlx.algoset.direction,
                  rlx.algoset.linesearch,
                  sts,
                  rpen)
end
