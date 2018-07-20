function InitializeMPCCActif(alas::ALASMPCC,
                             xj::Vector,
                             ρ::Vector,
                             usg::Vector,
                             ush::Vector,
                             uxl::Vector,
                             uxu::Vector,
                             ucl::Vector,
                             ucu::Vector)

 #Create an unconstrained penalized NLP with bounds
 pen_mpcc=CreatePenaltyNLP(alas,xj,ρ,usg,ush,uxl,uxu,ucl,ucu)

 #Create an ActifMPCC
 return ActifMPCC(pen_mpcc,
                  alas.r,alas.s,alas.t,
                  alas.mod.nb_comp,
                  alas.paramset,
                  alas.algoset.direction,
                  alas.algoset.linesearch)
end
