module RlxMPCCmod

import MPCCmod.MPCC, MPCCmod.obj, MPCCmod.grad, MPCCmod.grad!, MPCCmod.hess
import MPCCmod.cons_mp, MPCCmod.consG, MPCCmod.consH, MPCCmod.jacG, MPCCmod.jacH
import MPCCmod.cons_nl, MPCCmod.jac_nl, MPCCmod.viol_cons
import MPCCmod.jtprodG, MPCCmod.jtprodH, MPCCmod.jtprodnl
import MPCCmod.hessnl, MPCCmod.hessG, MPCCmod.hessH

import Relaxation.psi, Relaxation.phi, Relaxation.dphi

import NLPModels.AbstractNLPModel, NLPModels.NLPModelMeta, NLPModels.Counters

type RlxMPCC <: AbstractNLPModel

 meta     :: NLPModelMeta #A compléter
 counters :: Counters #ATTENTION : increment! ne marche pas?
 x0       :: Vector

 mod      :: MPCC
 r        :: Float64
 s        :: Float64
 t        :: Float64
 tb       :: Float64

 n        :: Int64 #dans le fond est optionnel si on a ncc
 ncc      :: Int64

end

function RlxMPCC(mod     :: MPCC,
                 r       :: Float64,
                 s       :: Float64,
                 t       :: Float64,
                 tb      :: Float64,
                 ncc     :: Int64,
                 n       :: Int64;
                 meta    :: NLPModelMeta = mod.mp.meta,
                 x       :: Vector = mod.mp.meta.x0)

println("Warning: RlxMPCC: il manque les meta plus précis du problème.")

 return RlxMPCC(meta,Counters(),x,mod,r,s,t,tb,n,ncc)
end


############################################################################
#
# Classical NLP functions on RlxMPCC
# obj, grad, grad!, hess, cons, cons!
#
############################################################################

function obj(rlxmpcc :: RlxMPCC, x :: Vector)
 return obj(rlxmpcc.mod, x)
end

function grad(rlxmpcc :: RlxMPCC, x :: Vector)
 return grad(rlxmpcc.mod, x)
end

function grad!(rlxmpcc :: RlxMPCC, x :: Vector, gx :: Vector)
 return grad!(rlxmpcc.mod, x, gx)
end

function hess(rlxmpcc :: RlxMPCC, x :: Vector)
 #renvoi la triangulaire inférieure tril(H,-1)'
 return hess(rlxmpcc.mod, x)
end

function cons(rlxmpcc :: RlxMPCC, x :: Vector)

 c  = cons_mp(rlxmpcc.mod, x)

 G  = consG(rlxmpcc.mod, x)
 H  = consH(rlxmpcc.mod, x)
 sc = vcat(G,H)

 cc = phi(G, H , rlxmpcc.r, rlxmpcc.s, rlxmpcc.t)

 return vcat(c,sc,cc)
end

#########################################################
#
# Return the violation of the constraints
# |yg - G(x)|, |yh-H(x)|, lc <= c(x) <= uc
#
#########################################################
function viol_cons_c(rlxmpcc :: RlxMPCC, x :: Vector)

 n, ncc = rlxmpcc.n, rlxmpcc.ncc

 if length(x) == n

  c = viol_cons_c(rlxmpcc.mod, x)

 elseif length(x) == n+2*ncc

  xn, yg, yh = x[1:n], x[n+1:n+ncc], x[n+ncc+1:n+2*ncc]

  c = viol_cons_c(rlxmpcc.mod, xn)
  G = consG(rlxmpcc.mod, xn)
  H = consH(rlxmpcc.mod, xn)

  c = vcat(abs.(yg-G),abs.(yh-H),c)

 else
  throw("error wrong dimension")
 end

 return c
end

#########################################################
#
# Return the violation of the constraints
# |yg - G(x)|, |yh-H(x)|, l <= x <= u, lc <= c(x) <= uc
#
#########################################################
function viol_cons_nl(rlxmpcc :: RlxMPCC, x :: Vector)

 n, ncc = rlxmpcc.n, rlxmpcc.ncc

 if length(x) == n

  c = viol_cons(rlxmpcc.mod, x)

 elseif length(x) == n+2*ncc

  xn, yg, yh = x[1:n], x[n+1:n+ncc], x[n+ncc+1:n+2*ncc]

  c = viol_cons(rlxmpcc.mod, xn)
  G = consG(rlxmpcc.mod, xn)
  H = consH(rlxmpcc.mod, xn)

  c = vcat(G-yg,H-yh,c)

 else
  throw("error wrong dimension")
 end

 return c
end

function viol_cons(rlxmpcc :: RlxMPCC, x :: Vector)

 n, ncc = rlxmpcc.n, rlxmpcc.ncc

 c = viol_cons_nl(rlxmpcc, x)

 if length(x) == n

  G = consG(rlxmpcc.mod, x)
  H = consH(rlxmpcc.mod, x)
  cc = max.(phi(G,H,rlxmpcc.r,rlxmpcc.s,rlxmpcc.t),0)
  sc = vcat(max.(rlxmpcc.mod.G.meta.lcon-G, 0),
            max.(rlxmpcc.mod.H.meta.lcon-H, 0))

 elseif length(x) == n+2*ncc

  xn, yg, yh = x[1:n], x[n+1:n+ncc], x[n+ncc+1:n+2*ncc]

  cc = max.(phi(yg,yh,rlxmpcc.r,rlxmpcc.s,rlxmpcc.t),0)
  sc = vcat(max.(rlxmpcc.mod.G.meta.lcon-yg, 0),
            max.(rlxmpcc.mod.H.meta.lcon-yh, 0))

 else
  throw("error wrong dimension")
 end

 return vcat(c,sc,cc)
end

function jac_nl(rlxmpcc :: RlxMPCC, x :: Vector)

  n = rlxmpcc.n

  Jc = jac_nl(rlxmpcc.mod, x)
  Jl, Ju = eye(n), eye(n)

  A = vcat(-Jl,Ju,-Jc,Jc)

 return A
end

function jac_slack(rlxmpcc :: RlxMPCC, x :: Vector)

 JG   = jacG(rlxmpcc.mod,x)
 JH   = jacH(rlxmpcc.mod,x)

 A = vcat(JG,JH)

 return A
end

function jac_nlslack(rlxmpcc :: RlxMPCC, x :: Vector)
 return vcat(jac_slack(rlxmpcc, x), jac_nl(rlxmpcc, x))
end

function jtprod_nlslack(rlxmpcc :: RlxMPCC, x :: Vector, v :: Vector)
#v of size 2*ncc + 2*n + 2*ncon

 n, ncc, ncon = rlxmpcc.n, rlxmpcc.ncc, rlxmpcc.mod.mp.meta.ncon
 xn = x[1:n]
 vbl, vbu = v[2*ncc+1:2*ncc+n], v[2*ncc+n+1:2*ncc+2*n]
 vlc, vuc = v[2*ncc+2*n+1:2*ncc+2*n+ncon], v[2*ncc+2*n+ncon+1:length(v)]

 return (- vbl + vbu
         - jtprodnl(rlxmpcc.mod, xn, vlc)
         + jtprodnl(rlxmpcc.mod, xn, vuc)
         + jtprodG(rlxmpcc.mod,xn,v[1:ncc]) 
         + jtprodH(rlxmpcc.mod,xn,v[ncc+1:2*ncc]))
end

function jac_rlx(rlxmpcc :: RlxMPCC, x :: Vector)

   JPHI = dphi(G,H,rlxmpcc.r,rlxmpcc.s,rlxmpcc.t)
   JPHIG, JPHIH = JPHI[1:rlxmpcc.ncc], JPHI[rlxmpcc.ncc+1:2*rlxmpcc.ncc]

   JG   = jacG(rlxmpcc.mod,x)
   JH   = jacH(rlxmpcc.mod,x)

   A = vcat(-JG, -JH, diagm(JPHIG)*JG + diagm(JPHIH)*JH)

 return A
end

function hess_nlslack(rlxmpcc :: RlxMPCC, x :: Vector, v :: Vector, objw :: Float64)

 n, ncc, ncon = rlxmpcc.n, rlxmpcc.ncc, rlxmpcc.mod.mp.meta.ncon
 xn = x[1:n]

 test= (hessnl(rlxmpcc.mod,xn,y=v[2*ncc+1:2*ncc+2*n+2*ncon], obj_weight = objw)
         + hessG(rlxmpcc.mod,xn,y=v[1:ncc], obj_weight = 0.0)
         + hessH(rlxmpcc.mod,xn,y=v[1+ncc:2*ncc], obj_weight = 0.0))

 return test
end

"""
Jacobienne des contraintes actives à precmpcc près
"""

function jac_actif(rlxmpcc :: RlxMPCC, x :: Vector, prec :: Float64)

  n = rlxmpcc.n

  Il = find(z->z<=prec,abs.(x-rlxmpcc.mod.mp.meta.lvar))
  Iu = find(z->z<=prec,abs.(x-rlxmpcc.mod.mp.meta.uvar))
  jl = zeros(n);jl[Il]=1.0;Jl=diagm(jl);
  ju = zeros(n);ju[Iu]=1.0;Ju=diagm(ju);

  IG=[];IH=[];IPHI=[];Ig=[];Ih=[];

 if rlxmpcc.mod.mp.meta.ncon+rlxmpcc.ncc ==0

  A=[]

 else

  c=cons_nl(rlxmpcc.mod,x)
  Ig=find(z->z<=prec,abs.(c-rlxmpcc.mod.mp.meta.lcon))
  Ih=find(z->z<=prec,abs.(c-rlxmpcc.mod.mp.meta.ucon))
  J = jac_nl(rlxmpcc.mod, x)
  Jg=J[Ig,1:n]
  Jh=J[Ih,1:n]

  if rlxmpcc.ncc>0

   G = consG(rlxmpcc.mod,x)
   H = consH(rlxmpcc.mod,x)
   IG   = find(z->z<=prec,abs.(G - rlxmpcc.mod.G.meta.lcon))
   IH   = find(z->z<=prec,abs.(H - rlxmpcc.mod.H.meta.lcon))
   IPHI = find(z->z<=prec,abs.(phi(G,H,rlxmpcc.r,rlxmpcc.s,rlxmpcc.t)))

   JPHI = dphi(G,H,rlxmpcc.r,rlxmpcc.s,rlxmpcc.t)
   JPHIG, JPHIH = JPHI[1:rlxmpcc.ncc][IPHI], JPHI[rlxmpcc.ncc+1:2*rlxmpcc.ncc][IPHI]

   JG   = jacG(rlxmpcc.mod,x)
   JH   = jacH(rlxmpcc.mod,x)

   A=[Jl;Ju;-Jg;Jh;-JG[IG,1:n];-JH[IH,1:n];diagm(JPHIG)*JG[IPHI,1:n] + diagm(JPHIH)*JH[IPHI,1:n]]'

  else

   A=[Jl;Ju;-Jg;Jh]'

  end
 end

 return A, Il,Iu,Ig,Ih,IG,IH,IPHI
end

#end of module
end
