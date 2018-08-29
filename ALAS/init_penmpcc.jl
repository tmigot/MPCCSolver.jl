function _initialize_pen_mpcc(rlx  :: RlxMPCCSolve,
                              xj   :: Vector,
                              ρ    :: Vector,
                              u    :: Vector)

 n     = rlx.nlp.n
 ncc   = rlx.nlp.ncc
 r,s,t = rlx.nlp.r, rlx.nlp.s, rlx.nlp.t

 if norm(xj[1:n] - rlx.xj[1:n], Inf) < eps(Float64)

  #we keep only the bound constraints
  feas = vcat(rlx.rrelax.feas[2*ncc+1:2*ncc+n],
              rlx.rrelax.feas[2*ncc+n+1:2*ncc+2*n],
              rlx.rrelax.feas_cc)

  #fx =  rlx.rrelax.fx #+ Penalty(rrelax.feas)

  rpen = RPen(xj, feas = feas)
 else
  rpen = RPen(xj)
 end

println("Warning: init_penmpcc: enlever pen_nlp")
 penmpcc = PenMPCC(xj,
                   rlx.nlp,
                   rlx.algoset.penalty,
                   r,s,t,
                   ρ,u,
                   ncc,n)

 return penmpcc, rpen
end
