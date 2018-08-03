function _slack_complementarity_projection(rlx  :: RlxMPCCSolve)

 ncc = rlx.mod.ncc
 n   = rlx.mod.n

 x = copy(rlx.xj)

 #projection sur les contraintes de bornes: l <= x <= u
 l = rlx.mod.mp.meta.lvar
 u = rlx.mod.mp.meta.uvar
 x[1:n] = x[1:n] + max.(l-x[1:n],0) + max.(x[1:n]-u,0)

 if ncc == 0 return x end

 #Initialization:
 if length(x) == n
  x = vcat(x, consG(x), consH(x))
 else
  x = x
 end

 #projection sur les contraintes de positivité relaxés : yG>=tb et yH>=tb
 x[n+1:n+ncc]           = max.(x[n+1:n+ncc], ones(ncc)*rlx.tb)
 x[n+ncc+1:n+2*ncc] = max.(x[n+ncc+1:n+2*ncc], ones(ncc)*rlx.tb)

 #projection sur les contraintes papillons : yG<=psi(yH,r,s,t) et yH<=psi(yG,r,s,t)
 for i = 1:ncc
  psiyg = psi(x[n+i],     rlx.r, rlx.s, rlx.t)
  psiyh = psi(x[n+ncc+i], rlx.r,rlx.s,rlx.t)

  if x[n+i]-psiyh > 0 && x[n+ncc+i] - psiyg > 0
   x[n+i] >= x[n+ncc+i] ? x[n+ncc+i] = psiyg : x[n+i] = psiyh
  end
 end

 return x
end
