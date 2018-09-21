module PenMPCCmod

#pas censer être là ?
import Relaxation.psi, Relaxation.phi, Relaxation.dphi

import RlxMPCCmod: RlxMPCC, jtprod_nlslack, hess_nlslack, jac_nlslack, viol_cons_nl, get_ncon, obj, grad, hess_relax

importall NLPModels
import NLPModels: AbstractNLPModel, NLPModelMeta, Counters, increment!

#############################################################################
#
# Problème pénalisé de la relaxation du MPCC
#
# min  f(x) + (\rho)^-1 P(x,cons(x),yg-G(x),yh-H(x))
# s.t. lvar <= (x,yg,yh) <= uvar
#      phi(yg,yh) <= 0
#
#############################################################################

#On a besoin de (r,s,t) pour les contraintes
type PenMPCC <: AbstractNLPModel

 meta     :: NLPModelMeta
 counters :: Counters
 x0       :: Vector

 rlx      :: RlxMPCC
 penalty  :: Function

 r        :: Float64
 s        :: Float64
 t        :: Float64

 ρ        :: Vector
 u        :: Vector

 n        :: Int64 #size of the original variable x
 ncc      :: Int64 #nb of complementarity constraints
 npen     :: Int64 #nb of penalized constraints

 function PenMPCC(x       :: Vector,
                  rlx     :: RlxMPCC,
                  penalty :: Function,
                  r       :: Float64,
                  s       :: Float64,
                  t       :: Float64,
                  ρ       :: Vector,
                  u       :: Vector,
                  ncc     :: Int64,
                  n       :: Int64)

  lvar = vcat(rlx.meta.lvar,
              rlx.mod.meta.lccG+rlx.tb*ones(ncc),
              rlx.mod.meta.lccH+rlx.tb*ones(ncc))
  uvar = vcat(rlx.meta.uvar,Inf*ones(ncc),Inf*ones(ncc))

  if length(lvar) != length(uvar) throw(error("Dimension error: bounds.")) end

  new_lcon = -Inf*ones(ncc)
  new_ucon = zeros(ncc)
  ncon = ncc

  nv = length(x)
  n = rlx.meta.nvar

  if nv != n + 2*ncc throw(error("Dimension error: initial point.")) end

  meta = NLPModelMeta(nv, x0 = x, lvar = lvar, uvar = uvar,
                                  ncon = ncon, 
                                  lcon = new_lcon, ucon = new_ucon)

  npen = 2*n + 2*get_ncon(rlx) + 2*ncc

  if (length(ρ) != npen) | (length(u) != npen) throw(error("Dimension error: penalty parameter.")) end

  return new(meta,Counters(),x,rlx,penalty,r,s,t,ρ,u,n,ncc,npen)
 end
end

function get_bounds(pen :: PenMPCC)

# tb  = pen.rlx.tb
# lvar = [pen.rlx.mod.meta.lvar;tb*ones(2*pen.ncc)] #devrait etre fait avec les mod.meta.lccG
# uvar = [pen.rlx.mod.meta.uvar;Inf*ones(2*pen.ncc)]

 return pen.meta.lvar, pen.meta.uvar
end

############################################################################
#
# Classical NLP functions on ActifMPCC
# obj, grad, grad!, hess, cons, cons!
#
############################################################################

function obj(pen :: PenMPCC, x :: Vector)

 n,ncc = pen.n, pen.ncc

 xn = x[1:n]
 err = viol_cons_nl(pen.rlx, x)
 hx,bx,gx = err[1:2*ncc], err[2*ncc+1:2*ncc+2*n], err[2*ncc+2*n+1:length(err)]

 fx = obj(pen.rlx,xn)
 px = pen.penalty(hx,bx,gx,pen.ρ,pen.u)

 return fx+px
end

function grad(pen_mpcc :: PenMPCC, x :: Vector)

 n,ncc = pen_mpcc.n, pen_mpcc.ncc

 err = viol_cons_nl(pen_mpcc.rlx, x)
 hx,bx,gx = err[1:2*ncc], err[2*ncc+1:2*ncc+2*n], err[2*ncc+2*n+1:length(err)]

 Jgx = Float64[]
 Jgx = pen_mpcc.penalty(hx,bx,gx,Jgx,pen_mpcc.ρ,pen_mpcc.u)

 gpx = vcat(jtprod_nlslack(pen_mpcc.rlx,x,Jgx[1:length(Jgx)-2*ncc]),
            -Jgx[length(Jgx)-2*ncc+1:length(Jgx)])
 gradx  = vcat(grad(pen_mpcc.rlx,x[1:n]),zeros(2*ncc))

 return gradx + gpx
end

function objgrad(pen_mpcc :: PenMPCC, x :: Vector)

 n,ncc = pen_mpcc.n, pen_mpcc.ncc
 xn = x[1:n]
 err = viol_cons_nl(pen_mpcc.rlx, x)
 hx,bx,gx = err[1:2*ncc], err[2*ncc+1:2*ncc+2*n], err[2*ncc+2*n+1:length(err)]

 fx = obj(pen_mpcc.rlx,xn)
 px = pen_mpcc.penalty(hx,bx,gx,pen_mpcc.ρ,pen_mpcc.u)

 Jgx = Float64[]
 Jgx = pen_mpcc.penalty(hx,bx,gx,Jgx,pen_mpcc.ρ,pen_mpcc.u)

 gpx = vcat(jtprod_nlslack(pen_mpcc.rlx,x,Jgx[1:length(Jgx)-2*ncc]),
            -Jgx[length(Jgx)-2*ncc+1:length(Jgx)])
 gradx  = vcat(grad(pen_mpcc.rlx,x[1:n]),zeros(2*ncc))

 return fx+px, gradx + gpx
end

function hess(pen        :: PenMPCC,
              x          :: Vector;
              obj_weight :: Float64 = 1.0,
              y          :: Vector = zeros(pen.meta.ncon))

 n,ncc = pen.n, pen.ncc

 if y != zeros 
  Tmp = hess_relax(pen.rlx, x[n+1:n+ncc], x[n+ncc+1:n+2*ncc], y)
 else
  Tmp = Float64[]
 end

 err = viol_cons_nl(pen.rlx, x)
 hx,bx,gx = err[1:2*ncc], err[2*ncc+1:2*ncc+2*n], err[2*ncc+2*n+1:length(err)]

 Jgx, Hx = Float64[], zeros(0,0)
 Jgx, Hx = pen.penalty(hx,bx,gx,Jgx,Hx,pen.ρ,pen.u)

 #la jacobienne des contraintes actives
 Jnls = vcat(ones(2*ncc),bx.>0.0,gx.>0.0) .* jac_nlslack(pen.rlx, x[1:n])

 rslt = hess_nlslack(pen.rlx, x[1:n], Jgx, 1.0) + Jnls' * Hx * Jnls

 rslt2 = Hx[1:2*ncc,1:2*ncc] + Tmp

 return cat([1,2],rslt,rslt2)
end

function hess_coord(pen        :: PenMPCC,
                    x          :: Vector;
                    obj_weight :: Float64 = 1.0,
                    y          :: Vector = zeros(pen.meta.ncon))
 return findnz(hess(pen, x, obj_weight = obj_weight, y = y))
end

#jacobienne des contraintes:
function jtprod(pen_mpcc :: PenMPCC, 
                x        :: Vector,
                lambda   :: Vector)

 #x of size n+2ncc, lambda of size
 n       = pen_mpcc.n
 ncc     = pen_mpcc.ncc
 r, s, t = pen_mpcc.r,pen_mpcc.s,pen_mpcc.t

 if length(x) != n+2*ncc || length(lambda) != 2*n+3*ncc return end

 uxl,uxu = lambda[1:n], lambda[1+n:2*n]
 usg,ush = lambda[2*n+1:2*n+ncc], lambda[2*n+ncc+1:2*n+2*ncc]
 uphi    = lambda[2*n+2*ncc+1:2*n+3*ncc]

 #bounds constraints on x
 Jn = uxu - uxl
 #constraints on the relaxed complementarity
 JPhi = dphi(x[n+1:n+ncc],x[n+ncc+1:n+2*ncc], r, s, t)

 if ncc == 1
  Js =  vcat(- usg,- ush) + JPhi * uphi[1] #un bug ?
 else
  Js =  vcat(- usg,- ush) + JPhi * uphi
 end

 return vcat(Jn,Js)
end

function jac(pen_mpcc :: PenMPCC, 
             x        :: Vector)

 #x of size n+2ncc, lambda of size
 n       = pen_mpcc.n
 ncc     = pen_mpcc.ncc
 r, s, t = pen_mpcc.r,pen_mpcc.s,pen_mpcc.t

 if length(x) != n+2*ncc throw(error("Dimension error: jac")) end

 #bounds constraints on x
 Jn = hcat(eye(n),-eye(n),zeros(n,3*ncc))

 #constraints on the relaxed complementarity
 JPhi = dphi(x[n+1:n+ncc],x[n+ncc+1:n+2*ncc], r, s, t)

 Js =  hcat(zeros(2*ncc,2*n),-eye(2*ncc), JPhi)

 return vcat(Jn,Js)
end

function jac_coord(pen :: PenMPCC, x :: Vector)
 return findnz(jac(pen, x))
end

function cons(pen_mpcc :: PenMPCC, x :: Vector)

 n   = pen_mpcc.n
 ncc = pen_mpcc.ncc
 r, s, t = pen_mpcc.r, pen_mpcc.s, pen_mpcc.t
 lvar, uvar = get_bounds(pen_mpcc)

 sg = x[n+1:n+ncc]
 sh = x[n+ncc+1:n+2*ncc]

 vlx = max.(- x[1:n] + lvar[1:n], 0)
 vux = max.(  x[1:n] - uvar[1:n], 0)

 vlg = max.(- sg + lvar[n+1:n+ncc], 0)
 vlh = max.(- sh + lvar[n+ncc+1:n+2*ncc], 0)

 vug = psi(sh, r, s, t) - sg
 vuh = psi(sg, r, s, t) - sh

 cx = vcat(vlx, vux, vlg, vlh, max.(vug.*vuh, 0))

 return cx
end

############################################################################
#LSQComputationMultiplier(pen::ActifMPCC,x::Vector,gradpen::Vector) :
#ma PenMPCC
#xj in n+2ncc
#gradpen in n+2ncc
#
#calcul la valeur des multiplicateurs de Lagrange pour la contrainte de complémentarité en utilisant moindre carré
############################################################################

function computation_multiplier_bool(pen     :: PenMPCC,
                                     gradpen :: Vector,
                                     xjk     :: Vector;
                                     prec    :: Float64 = eps(Float64))

   l = _computation_multiplier(pen, gradpen, xjk)

   l_negative = findfirst(x->x<0, l) != 0

 return l, l_negative
end

function _computation_multiplier(pen     :: PenMPCC,
                                 gradpen :: Vector,
                                 xj      :: Vector;
                                 prec    :: Float64 = eps(Float64))

 n     = pen.n
 ncc   = pen.ncc
 r,s,t = pen.r,pen.s,pen.t
 lvar, uvar = get_bounds(pen)

 sg = xj[n+1:n+ncc]
 sh = xj[n+ncc+1:n+2*ncc]
 x  = xj[1:n]

 dg =  dphi(sg,sh,r,s,t)
 phix = phi(sg,sh,r,s,t)

 gx = dg[1:ncc]
 gy = dg[ncc+1:2*ncc]

 wn1 = find(z->z<=prec,abs.(x-lvar[1:n]))
 wn2 = find(z->z<=prec,abs.(x-uvar[1:n]))
 w1  = find(z->z<=prec,abs.(sg-lvar[n+1:n+ncc]))
 w2  = find(z->z<=prec,abs.(sh-lvar[n+ncc+1:n+2*ncc]))
 wcomp = find(z->z<=prec,abs.(phix))

 #matrices des contraintes actives : (lx,ux,lg,lh,lphi)'*A=b
 nx1 = length(wn1)
 nx2 = length(wn2)
 nx  = nx1 + nx2

 nw1 = length(w1)
 nw2 = length(w2)
 nwcomp = length(wcomp)

 Dlg = -diagm(ones(nw1))
 Dlh = -diagm(ones(nw2))
 Dlphig = diagm(collect(gx)[wcomp])
 Dlphih = diagm(collect(gy)[wcomp])

 #matrix of size: nc+nw1+nw2+nwcomp x nc+nw1+nw2+2nwcomp
 A=[hcat(-diagm(ones(nx1)), zeros(nx1, nx2+nw1+nw2+2*nwcomp));
    hcat(zeros(nx2, nx1), diagm(ones(nx2)), zeros(nx2, nw1+nw2+2*nwcomp));
    hcat(zeros(nw1, nx), Dlg, zeros(nw1, nw2+2*nwcomp));
    hcat(zeros(nw2, nx), zeros(nw2, nw1), Dlh, zeros(nw2, 2*nwcomp));
    hcat(zeros(nwcomp, nx), zeros(nwcomp, nw1+nw2), Dlphig, Dlphih)]

 #vector of size: nc+nw1+nw2+2nwcomp
 b=-[gradpen[wn1];
     gradpen[wn2];
     gradpen[w1+n];
     gradpen[wcomp+n];
     gradpen[w2+n+ncc];
     gradpen[wcomp+n+ncc]] 

 #compute the multiplier using pseudo-inverse
 l = pinv(A')*b
 #l=A' \ b

 lk                  = zeros(2*n+3*ncc)
 lk[wn1]             = l[1:nx1]
 lk[n+wn2]           = l[nx1+1:nx]
 lk[2*n+w1]          = l[nx+1:nx+nw1]
 lk[2*n+ncc+w2]      = l[nx+nw1+1:nx+nw1+nw2]
 lk[2*n+2*ncc+wcomp] = l[nx+nw1+nw2+1:nx+nw1+nw2+nwcomp]

 return lk
end

#end of module
end
