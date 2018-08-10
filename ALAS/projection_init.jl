function _slack_complementarity_projection(rlx  :: RlxMPCCSolve)

 n     = rlx.nlp.n
 ncc   = rlx.nlp.ncc
 mod   = rlx.nlp.mod 
 tb    = rlx.nlp.tb
 r,s,t = rlx.nlp.r, rlx.nlp.s, rlx.nlp.t

 x = copy(rlx.xj)

 #projection sur les contraintes de bornes: l <= x <= u
 l = mod.mp.meta.lvar
 u = mod.mp.meta.uvar
 x[1:n] = x[1:n] + max.(l-x[1:n],0) + max.(x[1:n]-u,0)

 if ncc == 0 return x end

 #Initialization:
 if length(x) == n
  x = vcat(x, consG(x), consH(x))
 else
  x = x
 end

 #projection sur les contraintes de positivité relaxés : yG>=tb et yH>=tb
 x[n+1:n+ncc]           = max.(x[n+1:n+ncc], ones(ncc)*tb)
 x[n+ncc+1:n+2*ncc] = max.(x[n+ncc+1:n+2*ncc], ones(ncc)*tb)

 #projection sur les contraintes papillons : yG<=psi(yH,r,s,t) et yH<=psi(yG,r,s,t)
 for i = 1:ncc
  psiyg = psi(x[n+i],     r, s, t)
  psiyh = psi(x[n+ncc+i], r, s, t)

  if x[n+i]-psiyh > 0 && x[n+ncc+i] - psiyg > 0
   x[n+i] >= x[n+ncc+i] ? x[n+ncc+i] = psiyg : x[n+i] = psiyh
  end
 end

 return x
end
