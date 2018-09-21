module RlxMPCCmod

import MPCCmod: MPCC, obj, grad, grad!, hess, viol_mp, consG, consH, jacG, jacH, cons_nl, jac_nl, viol_mp, jtprodG, jtprodH, jtprodnl, hessnl, hessG, hessH

import Relaxation: psi, phi, dphi, ddphi

importall NLPModels
import NLPModels: AbstractNLPModel, NLPModelMeta, Counters, increment!

"""
Definit le type RlxMPCC :
min 	f(x)
s.t. 	l <= x <= u
	lcon(tb) <= cnl(x) <= ucon

with

cnl(x) := c(x),G(x),H(x),Phi(G(x),H(x),t)

Some functions are compatible with slack variables:

min 	f(x)
s.t.	0 = G(x)-yg
	0 = H(x)-yh
	l <= x <= u
	lcon(tb) <= cnl_slack(x) <= ucon
with

cnl_slack(x) := c(x),yg,yh,Phi(yg,yh,t)
"""

type RlxMPCC <: AbstractNLPModel

 meta     :: NLPModelMeta
 counters :: Counters

 x0       :: Vector

 mod      :: MPCC
 r        :: Float64
 s        :: Float64
 t        :: Float64
 tb       :: Float64

 n        :: Int64 #better use meta.nvar
 ncc      :: Int64


 function RlxMPCC(mod     :: MPCC,
                  r       :: Float64,
                  s       :: Float64,
                  t       :: Float64,
                  tb      :: Float64;
                  meta    :: NLPModelMeta = mod.mp.meta,
                  x       :: Vector = mod.meta.x0)

  #meta is given without slacks
  ncc = mod.meta.ncc

  n = mod.meta.nvar
  ncon = mod.meta.ncon + 3*ncc

  new_lcon = vcat(mod.meta.lcon,
                  mod.meta.lccG+tb*ones(ncc),
                  mod.meta.lccH+tb*ones(ncc),
                  -Inf*ones(ncc))
  new_ucon = vcat(mod.meta.ucon,
                  Inf*ones(2*ncc),
                  zeros(ncc))

  meta = NLPModelMeta(n, x0 = x, lvar = mod.meta.lvar, uvar = mod.meta.uvar,
                                 ncon = ncon, 
                                 lcon = new_lcon, ucon = new_ucon)

  if tb > 0.0 throw(error("Domain error: tb must be non-positive")) end
  if minimum([r,s,t])<0.0 
   throw(error("Domain error: (r,s,t) must be non-negative"))
  end

  return new(meta,Counters(),x,mod,r,s,t,tb,n,ncc)
 end
end

NotImplemented(args...; kwargs...) = throw(NotImplementedError(""))

#return the number of "classical" non-linear constraints
function get_ncon(rlxmpcc :: RlxMPCC) return rlxmpcc.meta.ncon - 3*rlxmpcc.ncc end

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

function objgrad(rlxmpcc :: RlxMPCC, x :: Vector)
 return obj(rlxmpcc, x), grad(rlxmpcc, x)
end

function objgrad!(rlxmpcc :: RlxMPCC, x :: Vector, gx :: Vector)
 return obj(rlxmpcc, x), grad!(rlxmpcc, x, gx)
end

function hess(rlxmpcc :: RlxMPCC, x :: Vector; obj_weight = 1.0, y = zeros)

 if y != zeros
  return hess(rlxmpcc, x, obj_weight, y)
 else
  return hess(rlxmpcc.mod, x)
 end
end

function hess(rlxmpcc :: RlxMPCC, x :: Vector, obj_weight :: Float64, y :: Vector)

 ncc = rlxmpcc.ncc
 nl = get_ncon(rlxmpcc)

 y_nl, y_G, y_H, y_Phi = y[1:nl], y[nl+1:nl+ncc], y[nl+ncc+1:nl+2*ncc], y[nl+2*ncc+1:nl+3*ncc]

 hess_mp = hess(rlxmpcc.mod, x, obj_weight = obj_weight, y = y_nl)
 if ncc > 0

  G, H = consG(rlxmpcc.mod, x), consH(rlxmpcc.mod, x)
  dgg, dgh, dhh = ddphi(G, H, rlxmpcc.r, rlxmpcc.s, rlxmpcc.t)
  dph = dphi(G, H, rlxmpcc.r, rlxmpcc.s, rlxmpcc.t)
  y_G += dph[1:ncc] 
  hess_G = hessG(rlxmpcc.mod, x, obj_weight = obj_weight, y = y_G)
  y_H += dph[ncc+1:2*ncc]
  hess_H = hessH(rlxmpcc.mod, x, obj_weight = obj_weight, y = y_H)
  jG, jH = jacG(rlxmpcc.mod, x), jacH(rlxmpcc.mod, x)
  hess_P = jG' * (diagm(dgg) * jG) + jH' * (diagm(dgh) * jH) + jG' * (diagm(dgh) * jH) + jH' * (diagm(dgh) * jG)

 else
  hess_G, hess_H, hess_P = Float64[], Float64[], Float64[]
 end

 return tril(hess_mp + hess_G + hess_H + hess_P)
end

function hess_nlslack(rlxmpcc :: RlxMPCC, x :: Vector, v :: Vector, objw :: Float64)

 n, ncc, ncon = rlxmpcc.meta.nvar, rlxmpcc.ncc, rlxmpcc.mod.meta.ncon
 xn = x[1:n]

 test = (hessnl(rlxmpcc.mod,xn,y=v[2*ncc+1:2*ncc+2*n+2*ncon], obj_weight = objw)
         + hessG(rlxmpcc.mod,xn,y=v[1:ncc], obj_weight = 0.0)
         + hessH(rlxmpcc.mod,xn,y=v[1+ncc:2*ncc], obj_weight = 0.0))

 return test
end

function hess_relax(rlxmpcc :: RlxMPCC, yg :: Vector, yh :: Vector, y :: Vector)
 ncc = rlxmpcc.ncc

 if length(y) == 3*ncc
  y_Phi = y[2*ncc+1:3*ncc]
 elseif length(y) == ncc
  y_Phi = y
 else
  throw(error("Dimension error: y"))
 end

 if ncc > 0

  dgg, dgh, dhh = ddphi(yg, yh, rlxmpcc.r, rlxmpcc.s, rlxmpcc.t)
  hess_P = vcat(hcat(diagm(y_Phi.*dgg),diagm(y_Phi.*dgh)),
                hcat(diagm(y_Phi.*dgh),diagm(y_Phi.*dhh)))

 else
  hess_P = Float64[]
 end

 return hess_P #matrix of size 2*ncc x 2*ncc
end

function hess_coord(rlxmpcc    :: RlxMPCC,
                    x          :: Vector;
                    obj_weight :: Float64 = 1.0,
                    y          :: Vector = zeros(rlxmpcc.meta.ncon))
 return findnz(hess(rlxmpcc, x, obj_weight = obj_weight, y = y))
end

#########################################################
#
# Return the vector of the constraints
# |yg - G(x)|, |yh-H(x)|, c(x), yG, yH, Phi(yg,yh,t)
#
#########################################################
function cons(rlxmpcc :: RlxMPCC, x :: Vector)

 n, ncc = rlxmpcc.meta.nvar, rlxmpcc.ncc

 if length(x) == n

  c  = cons_nl(rlxmpcc.mod, x) #c(x)
  G  = consG(rlxmpcc.mod, x)
  H  = consH(rlxmpcc.mod, x)
  sc = vcat(G,H)
  cc = phi(G, H, rlxmpcc.r, rlxmpcc.s, rlxmpcc.t)

 elseif length(x) == n + 2*ncc

  xn, yg, yh = x[1:n], x[n+1:n+ncc], x[n+ncc+1:n+2*ncc]

  G  = consG(rlxmpcc.mod, x)
  H  = consH(rlxmpcc.mod, x)
  c  = vcat(G-yg,H-yh,cons_nl(rlxmpcc.mod, x))
  sc = vcat(yg, yh)
  cc = phi(yg,yh,rlxmpcc.r,rlxmpcc.s,rlxmpcc.t)

 else

  throw(error("Domain error"))

 end

 return vcat(c,sc,cc)
end

#########################################################
#
# Return the violation of the constraints
# |yg - G(x)|, |yh-H(x)|
# lx <= x <= ux
# lc <= c(x) <= uc
# 0 <= yg, 0 <= yh
# Phi(yg,yh,t) <= 0
#
#########################################################
function viol(rlxmpcc :: RlxMPCC, x :: Vector)

 n, ncc = rlxmpcc.meta.nvar, rlxmpcc.ncc

 if length(x) == n

  c = cons(rlxmpcc, x)

  viol_slack = Float64[]
  viol_x = max.(rlxmpcc.meta.lvar-x, 0)+max.(x-rlxmpcc.meta.uvar, 0)
  viol_c = max.(rlxmpcc.meta.lcon-c, 0)+max.(c-rlxmpcc.meta.ucon, 0)

 elseif length(x) == n + 2*ncc

  c = cons(rlxmpcc, x)

  c_slack = c[1:2*ncc]
  viol_slack = abs.(c_slack)
  xn = x[1:n]
  viol_x = max.(rlxmpcc.meta.lvar-xn, 0)+max.(xn-rlxmpcc.meta.uvar, 0)
  c_nl = c[2*ncc+1:length(c)]
  viol_c = max.(rlxmpcc.meta.lcon-c_nl, 0)+max.(c_nl-rlxmpcc.meta.ucon, 0)

 else

  throw(error("Domain error"))

 end

 return vcat(viol_slack,viol_x, viol_c)
end

#########################################################
#
# Return the violation of the "classical" constraints
# |yg - G(x)|, |yh-H(x)|, l <= x <= u, lc <= c(x) <= uc
#
#########################################################
function viol_cons_nl(rlxmpcc :: RlxMPCC, x :: Vector)

 n, ncc = rlxmpcc.meta.nvar, rlxmpcc.ncc

 if length(x) == n

  c = viol_mp(rlxmpcc.mod, x)

 elseif length(x) == n+2*ncc

  xn, yg, yh = x[1:n], x[n+1:n+ncc], x[n+ncc+1:n+2*ncc]

  c = viol_mp(rlxmpcc.mod, xn)
  G = consG(rlxmpcc.mod, xn)
  H = consH(rlxmpcc.mod, xn)

  c = vcat(G-yg,H-yh,c)

 else
  throw(error("error wrong dimension"))
 end

 return c
end

function jac(rlxmpcc :: RlxMPCC, x :: Vector)
 A, Il,Iu,Ig,Ih,IG,IH,IPHI = jac_actif(rlxmpcc, x, 0.0)
 return A
end

function jac_coord(rlxmpcc :: RlxMPCC, x :: Vector)
 return findnz(jac(rlxmpcc, x))
end

function jac_nl(rlxmpcc :: RlxMPCC, x :: Vector)

 n = rlxmpcc.meta.nvar
 ncon = get_ncon(rlxmpcc)

 Jl, Ju = eye(n), eye(n)

 if ncon > 0
  Jc = jac_nl(rlxmpcc.mod, x)
  A = vcat(-Jl,Ju,-Jc,Jc)
 else
  A = vcat(-Jl,Ju)
 end

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

 n, ncc, ncon = rlxmpcc.meta.nvar, rlxmpcc.ncc, rlxmpcc.mod.meta.ncon
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

###########################################################################
#
# jac_actif(rlxmpcc :: RlxMPCC, x :: Vector, prec :: Float64)
# return A, Il,Iu,Ig,Ih,IG,IH,IPHI
#
###########################################################################
include("rlxmpcc_jacactif.jl")

#end of module
end
