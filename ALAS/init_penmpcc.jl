function InitializePenMPCC(alas :: ALASMPCC,
                           xj   :: Vector,
                           ρ    :: Vector,
                           u    :: Vector)

 n       = alas.mod.n
 nb_comp = alas.mod.nb_comp

 #Create a penalized NLP with bounds
 pen_nlp = CreatePenaltyNLP(alas, xj, ρ, u)

 if norm(xj[1:n] - alas.xj[1:n], Inf) < eps(Float64)

  #we keep only the bound constraints
  feas = vcat(alas.rrelax.feas[2*nb_comp+1:2*nb_comp+n],
              alas.rrelax.feas[2*nb_comp+n+1:2*nb_comp+2*n],
              alas.rrelax.feas_cc)

  #fx =  alas.rrelax.fx #+ Penalty(rrelax.feas)

  rpen = RPen(xj, feas = feas)
 else
  repen = RPen(xj)
 end

 penmpcc = PenMPCC(pen_nlp,
                   alas.r,alas.s,alas.t,
                   ρ,u,
                   alas.mod.nb_comp,alas.mod.n)

 return penmpcc,rpen
end
