function PenMPCCSolve(pen        :: PenMPCC,
                      x          :: Vector, #long
                      ncc        :: Int64,
                      paramset   :: ParamSet,
                      algoset    :: AlgoSet,
                      spen       :: StoppingPen,
                      rpen       :: RPen)

 r,s,t = pen.r,pen.s,pen.t
 lvar, uvar = get_bounds(pen)
 meta = pen.meta

 nn  = length(x) #n + 2ncc
 n   = length(x)-2*ncc
 xk  = x[1:n]
 ygk = x[n+1:n+ncc]
 yhk = x[n+ncc+1:n+2*ncc]

 w   = zeros(Bool,nn,2)
 #active bounds
 w[1:n+ncc,1]   = x[1:n+ncc]          .== lvar[1:n+ncc]
 w[n+1:n+ncc,2] = x[n+ncc+1:n+2*ncc]  .== lvar[n+ncc+1:n+2*ncc]
 w[1:n,2]       = xk                  .== uvar[1:n]

  # puis la boucle: est-ce qu'il y a Relaxation.psi vectoriel ?
  #A simplifier
  for l=1:ncc
   if ygk[l]==psi(yhk[l],r,s,t)
    w[n+l+ncc,1]=true;
   end
   if yhk[l]==psi(ygk[l],r,s,t)
    w[n+l+ncc,2]=true;
   end
  end

 wnc = find(.!w[1:n,1] .& .!w[1:n,2])
 wn1 = find(w[1:n,1])
 wn2 = find(w[1:n,2])

 w1  = find(w[n+1:n+ncc,1])
 w2  = find(w[n+1:n+ncc,2])
 w3  = find(w[n+ncc+1:n+2*ncc,1])
 w4  = find(w[n+ncc+1:n+2*ncc,2])

 wcomp = find(w[n+ncc+1:n+2*ncc,1] .| w[n+ncc+1:n+2*ncc,2])
 w13c  = find(.!w[n+1:n+ncc,1] .& .!w[n+ncc+1:n+2*ncc,1])
 w24c  = find(.!w[n+1:n+ncc,2] .& .!w[n+ncc+1:n+2*ncc,2])

 wc    = find(.!w[n+1:n+ncc,1] .& .!w[n+1:n+ncc,2] .& .!w[n+ncc+1:n+2*ncc,1] .& .!w[n+ncc+1:n+2*ncc,2])
 wcc   = find((w[n+1:n+ncc,1] .| w[n+ncc+1:n+2*ncc,1]) .& (w[n+1:n+ncc,2] .| w[n+ncc+1:n+2*ncc,2]))

 wnew = zeros(Bool,0,0)
 dj = zeros(n+2*ncc)

 crho = 1.0
 beta = 0.0
 Hess = eye(n+2*ncc)

 return PenMPCCSolve(meta,Counters(),x,pen,w,n,ncc,
                     wnc,wn1,wn2,w1,w2,w3,w4,
                     wcomp,w13c,w24c,wc,wcc,wnew,dj,crho,beta,Hess,
                     paramset,algoset,rpen,spen)
end
