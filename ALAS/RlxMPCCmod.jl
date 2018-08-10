module RlxMPCCmod

import MPCCmod.MPCC, MPCCmod.obj, MPCCmod.grad, MPCCmod.grad!, MPCCmod.hess
import MPCCmod.cons_mp, MPCCmod.consG, MPCCmod.consH, MPCCmod.jacG, MPCCmod.jacH
import MPCCmod.cons_nl, MPCCmod.jac_nl

import Relaxation.psi, Relaxation.phi, Relaxation.dphi
import ParamSetmod.ParamSet

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

 c = cons_mp(rlxmpcc.mod, x)

 G = consG(rlxmpcc.mod, x)
 H = consH(rlxmpcc.mod, x)
 cc = phi(G,H,rlxmpcc.r,rlxmpcc.s,rlxmpcc.t)

 return vcat(c,sc,cc)
end

function viol_cons(rlxmpcc :: RlxMPCC, x :: Vector)

 c = viol_cons(rlxmpcc.mod, x)

 G = consG(rlxmpcc.mod, x)
 H = consH(rlxmpcc.mod, x)
 cc = max.(phi(G,H,rlxmpcc.r,rlxmpcc.s,rlxmpcc.t),0)
 sc = vcat(max.(rlxmpcc.mod.G.meta.lvar-G, 0),
           max.(rlxmpcc.mod.H.meta.lvar-H, 0))

 return vcat(c,sc,cc)
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
