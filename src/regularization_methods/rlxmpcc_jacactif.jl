"""
Jacobienne des contraintes actives à precmpcc près
"""
function jac_actif(rlxmpcc :: RlxMPCC, x :: Vector, prec :: Float64)

  n = rlxmpcc.n
  ncc = rlxmpcc.ncc
  r, s, t, tb = rlxmpcc.r, rlxmpcc.s, rlxmpcc.t, rlxmpcc.tb
  mod = rlxmpcc.mod

  Il = findall(z->z<=prec,abs.(x - mod.meta.lvar))
  Iu = findall(z->z<=prec,abs.(x - mod.meta.uvar))
  jl = zeros(n);jl[Il].=1.0;Jl=diagm(0 => jl);
  ju = zeros(n);ju[Iu].=1.0;Ju=diagm(0 => ju);

  IG=[];IH=[];IPHI=[];Ig=[];Ih=[];

 if mod.meta.ncon+ncc ==0

  A=[]

 else

  if mod.meta.ncon > 0
   c = cons_nl(mod, x)
   J = jac_nl(mod, x)
  else
   c = Float64[]
   J = sparse(zeros(0,2))
  end

  Ig=findall(z->z<=prec,abs.(c- mod.meta.lcon))
  Ih=findall(z->z<=prec,abs.(c- mod.meta.ucon))

  Jg, Jh = zeros(mod.meta.ncon,n), zeros(mod.meta.ncon,n)

  Jg[Ig,1:n] = J[Ig,1:n]
  Jh[Ih,1:n] = J[Ih,1:n]

  if ncc>0

   G = consG(mod,x)
   H = consH(mod,x)
   IG   = findall(z->z<=prec,abs.(G - tb*ones(ncc) - mod.meta.lccG))
   IH   = findall(z->z<=prec,abs.(H - tb*ones(ncc) - mod.meta.lccH))
   IPHI = findall(z->z<=prec,abs.(phi(G, H, r, s, t)))

   JPHI = dphi(G,H,rlxmpcc.r,rlxmpcc.s,rlxmpcc.t)
   JPHIG, JPHIH = JPHI[1:ncc][IPHI], JPHI[ncc+1:2*ncc][IPHI]

   JG   = jacG(mod,x)
   JH   = jacH(mod,x)

   JBG, JBH = zeros(ncc, n), zeros(ncc, n)
   JBG[IG,1:n] = JG[IG,1:n]
   JBH[IH,1:n] = JH[IH,1:n]

   JP = zeros(ncc, n)
   JP[IPHI,1:n] = diagm(0 => JPHIG)*JG[IPHI,1:n] + diagm(0 => JPHIH)*JH[IPHI,1:n]

   A=[Jl;Ju;-Jg;Jh;-JBG;-JBH;JP]'

  else

   A=[Jl;Ju;-Jg;Jh]' #c'est à l'envers!

  end
 end

 return A, Il,Iu,Ig,Ih,IG,IH,IPHI
end

function jac_actif2(rlxmpcc :: RlxMPCC, x :: Vector, prec :: Float64)

  n = rlxmpcc.n
  ncc = rlxmpcc.ncc
  r, s, t, tb = rlxmpcc.r, rlxmpcc.s, rlxmpcc.t, rlxmpcc.tb
  mod = rlxmpcc.mod

  IG=[];IH=[];IPHI=[];Ig=[];Ih=[];

 if mod.meta.ncon+ncc ==0

  A=[]

 else

  if mod.meta.ncon > 0
   c = cons_nl(mod, x)
   J = jac_nl(mod, x)
  else
   c = Float64[]
   J = sparse(zeros(0,2))
  end

  Ig=findall(z->z<=prec,abs.(c- mod.meta.lcon))
  Ih=findall(z->z<=prec,abs.(c- mod.meta.ucon))

  Jg, Jh = zeros(mod.meta.ncon,n), zeros(mod.meta.ncon,n)

  Jg[Ig,1:n] = J[Ig,1:n]
  Jh[Ih,1:n] = J[Ih,1:n]

  if ncc>0

   G = consG(mod,x)
   H = consH(mod,x)

   JPHI = dphi(G,H,rlxmpcc.r,rlxmpcc.s,rlxmpcc.t)
   JPHIG, JPHIH = JPHI[1:ncc], JPHI[ncc+1:2*ncc]

   JG   = jacG(mod,x)
   JH   = jacH(mod,x)
   JP = diagm(0 => JPHIG)*JG + diagm(0 => JPHIH)*JH

   A=[J;JG;JH;JP]

  else

   A=J

  end
 end

 return A
end
