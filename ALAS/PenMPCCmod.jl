module PenMPCCmod

import Relaxation.psi, Relaxation.phi, Relaxation.dphi
import ParamSetmod.ParamSet

import NLPModels.grad!, NLPModels.hess, NLPModels.obj, NLPModels.grad
import NLPModels.AbstractNLPModel, NLPModels.NLPModelMeta, NLPModels.Counters



type PenMPCC <: AbstractNLPModel

 meta     :: NLPModelMeta
 counters :: Counters #ATTENTION : increment! ne marche pas?
 x0       :: Vector

 nlp      :: AbstractNLPModel
 r        :: Float64
 s        :: Float64
 t        :: Float64

 ρ        :: Vector
 u        :: Vector

 n        :: Int64 #dans le fond est optionnel si on a nb_comp
 nb_comp  :: Int64

end

function PenMPCC(nlp     :: AbstractNLPModel,
                 r       :: Float64,
                 s       :: Float64,
                 t       :: Float64,
                 ρ       :: Vector,
                 u       :: Vector,
                 nb_comp :: Int64,
                 n       :: Int64)

 
 meta = nlp.meta
 x    = nlp.meta.x0


 return PenMPCC(meta,Counters(),x,nlp,r,s,t,ρ,u,n,nb_comp)
end


############################################################################
#
# Classical NLP functions on ActifMPCC
# obj, grad, grad!, hess, cons, cons!
#
############################################################################

function obj(pen_mpcc :: PenMPCC, x :: Vector)
 return obj(pen_mpcc.nlp, x)
end

function grad(pen_mpcc :: PenMPCC, x :: Vector)
 return grad(pen_mpcc.nlp, x)
end

function grad!(pen_mpcc :: PenMPCC, x :: Vector, gx :: Vector)
 return grad!(pen_mpcc.nlp, x, gx)
end

function hess(pen_mpcc :: PenMPCC, x :: Vector)
 #renvoi la triangulaire inférieure tril(H,-1)'
 return hess(pen_mpcc.nlp, x)
end

#jacobienne des contraintes:

function jac(pen_mpcc :: PenMPCC, x :: Vector, lambda :: Vector)
 #x of size n+2nb_comp, lambda of size
 n = pen_mpcc.n
 nb_comp = pen_mpcc.nb_comp
 if length(x) != n+2*nb_comp || length(lambda) != 2*n+3*nb_comp return end

 uxl,uxu = lambda[1:n], lambda[1+n:2*n]
 usg,ush = lambda[2*n+1:2*n+nb_comp], lambda[2*n+nb_comp+1:2*n+2*nb_comp]
 uphi = lambda[2*n+2*nb_comp+1:2*n+3*nb_comp]

 #bounds constraints on x
 Jn = uxu - uxl
 #constraints on the relaxed complementarity
 JPhi = dphi(x[n+1:n+nb_comp],x[n+nb_comp+1:n+2*nb_comp],pen_mpcc.r,pen_mpcc.s,pen_mpcc.t)
 if nb_comp == 1
  Js =  vcat(- usg,- ush) + JPhi * uphi[1]
 else
  Js =  vcat(- usg,- ush) + JPhi * uphi
 end
 return vcat(Jn,Js)
end

function cons(pen_mpcc :: PenMPCC, x :: Vector)

 n       = pen_mpcc.n
 nb_comp = pen_mpcc.nb_comp

 sg = x[n+1:n+nb_comp]
 sh = x[n+nb_comp+1:n+2*nb_comp]

 vlx = max.(- x[1:n] + pen_mpcc.nlp.meta.lvar[1:n], 0)
 vux = max.(  x[1:n] - pen_mpcc.nlp.meta.uvar[1:n], 0)

 vlg = max.(- sg + pen_mpcc.nlp.meta.lvar[n+1:n+nb_comp], 0)
 vlh = max.(- sh + pen_mpcc.nlp.meta.lvar[n+nb_comp+1:n+2*nb_comp], 0)

 vug = psi(sh, pen_mpcc.r, pen_mpcc.s, pen_mpcc.t) - sg
 vuh = psi(sg, pen_mpcc.r, pen_mpcc.s, pen_mpcc.t) - sh

 cx = vcat(vlx, vux, vlg, vlh, max.(vug.*vuh, 0))

 return cx
end

############################################################################
#LSQComputationMultiplier(pen::ActifMPCC,x::Vector,gradpen::Vector) :
#ma PenMPCC
#xj in n+2nb_comp
#gradpen in n+2nb_comp
#
#calcul la valeur des multiplicateurs de Lagrange pour la contrainte de complémentarité en utilisant moindre carré
############################################################################

function ComputationMultiplierBool(pen     :: PenMPCC,
                                   gradpen :: Vector,
                                   xjk     :: Vector;
                                   prec    :: Float64 = eps(Float64))

   l = ComputationMultiplier(pen, gradpen, xjk)

   l_negative = findfirst(x->x<0, l) != 0

 return l, l_negative
end

function ComputationMultiplier(pen     :: PenMPCC,
                               gradpen :: Vector,
                               xj      :: Vector;
                               prec    :: Float64 = eps(Float64))

 n       = pen.n
 nb_comp = pen.nb_comp

 sg = xj[n+1:n+nb_comp]
 sh = xj[n+nb_comp+1:n+2*nb_comp]
 x  = xj[1:n]

 dg = dphi(sg,sh,pen.r,pen.s,pen.t)
 phix = phi(sg,sh,pen.r,pen.s,pen.t)

 gx = dg[1:nb_comp]
 gy = dg[nb_comp+1:2*nb_comp]

 wn1 = find(z->z<=prec,abs.(x-pen.nlp.meta.lvar[1:n]))
 wn2 = find(z->z<=prec,abs.(x-pen.nlp.meta.uvar[1:n]))
 w1  = find(z->z<=prec,abs.(sg-pen.nlp.meta.lvar[n+1:n+nb_comp]))
 w2  = find(z->z<=prec,abs.(sh-pen.nlp.meta.lvar[n+nb_comp+1:n+2*nb_comp]))
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
     gradpen[w2+n+nb_comp];
     gradpen[wcomp+n+nb_comp]] 

 #compute the multiplier using pseudo-inverse
 l = pinv(A')*b
 #l=A' \ b

 lk                      = zeros(2*n+3*nb_comp)
 lk[wn1]                 = l[1:nx1]
 lk[n+wn2]               = l[nx1+1:nx]
 lk[2*n+w1]              = l[nx+1:nx+nw1]
 lk[2*n+nb_comp+w2]      = l[nx+nw1+1:nx+nw1+nw2]
 lk[2*n+2*nb_comp+wcomp] = l[nx+nw1+nw2+1:nx+nw1+nw2+nwcomp]

 return lk
end

#end of module
end
