function InitializeSolvePenMPCC(alas    :: ALASMPCC,
                                penmpcc :: PenMPCC,
                                rpen    :: RPen,
                                xj      :: Vector)

 sts    = TStopping(max_iter=alas.paramset.ite_max_viol, atol=alas.prec)

 #Create an ActifMPCC
 return ActifMPCC(penmpcc,
                  penmpcc.nb_comp,
                  alas.paramset,
                  alas.algoset.direction,
                  alas.algoset.linesearch,
                  sts,
                  rpen)
end
