"""
Jacobienne des contraintes actives à precmpcc près
"""
function jac_actif(rlxmpcc :: RlxMPCC, x :: Vector, prec :: Float64)

  n = rlxmpcc.n
  ncc = rlxmpcc.ncc

  Il = find(z->z<=prec,abs.(x-rlxmpcc.mod.meta.lvar))
  Iu = find(z->z<=prec,abs.(x-rlxmpcc.mod.meta.uvar))
  jl = zeros(n);jl[Il]=1.0;Jl=diagm(jl);
  ju = zeros(n);ju[Iu]=1.0;Ju=diagm(ju);

  IG=[];IH=[];IPHI=[];Ig=[];Ih=[];

 if rlxmpcc.mod.meta.ncon+ncc ==0

  A=[]

 else

  if mod.meta.ncon > 0
   c = cons_nl(rlxmpcc.mod, x)
   J = jac_nl(rlxmpcc.mod, x)
  else
   c = Float64[]
   J = sparse(zeros(0,2))
  end

  Ig=find(z->z<=prec,abs.(c-rlxmpcc.mod.meta.lcon))
  Ih=find(z->z<=prec,abs.(c-rlxmpcc.mod.meta.ucon))

  Jg, Jh = zeros(rlxmpcc.mod.meta.ncon,n), zeros(rlxmpcc.mod.meta.ncon,n)

  Jg[Ig,1:n] = J[Ig,1:n]
  Jh[Ih,1:n] = J[Ih,1:n]

  if ncc>0

   G = consG(rlxmpcc.mod,x)
   H = consH(rlxmpcc.mod,x)
   IG   = find(z->z<=prec,abs.(G - tb*ones(ncc) - rlxmpcc.mod.meta.lccG))
   IH   = find(z->z<=prec,abs.(H - tb*ones(ncc) - rlxmpcc.mod.meta.lccH))
   IPHI = find(z->z<=prec,abs.(phi(G,H,rlxmpcc.r,rlxmpcc.s,rlxmpcc.t)))

   JPHI = dphi(G,H,rlxmpcc.r,rlxmpcc.s,rlxmpcc.t)
   JPHIG, JPHIH = JPHI[1:ncc][IPHI], JPHI[ncc+1:2*ncc][IPHI]

   JG   = jacG(rlxmpcc.mod,x)
   JH   = jacH(rlxmpcc.mod,x)

   JBG, JBH = zeros(ncc, n), zeros(ncc, n)
   JBG[IG,1:n] = JG[IG,1:n]
   JBH[IH,1:n] = JH[IH,1:n]

   JP = zeros(ncc, n)
   JP[IPHI,1:n] = diagm(JPHIG)*JG[IPHI,1:n] + diagm(JPHIH)*JH[IPHI,1:n]

   A=[Jl;Ju;-Jg;Jh;-JBG;-JBH;JP]'

  else

   A=[Jl;Ju;-Jg;Jh]'

  end
 end

 return A, Il,Iu,Ig,Ih,IG,IH,IPHI
end
