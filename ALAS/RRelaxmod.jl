module RRelaxmod

import MPCCmod.MPCC, MPCCmod.obj, MPCCmod.grad
import MPCCmod.consG, MPCCmod.consH
import MPCCmod.viol_contrainte

import Relaxation.phi

import RPenmod.RPen

import RlxMPCCmod.RlxMPCC

type RRelax

 x              :: Vector
 fx             :: Float64 #objective at x
 gx             :: Vector #gradient at x

 feas           :: Vector #violation of the constraints
 feas_cc        :: Vector #violation of the relaxed constraints
 norm_feas      :: Float64 # norm of feas

 dual_feas      :: Vector
 norm_dual_feas :: Float64

 lambda         :: Vector
 lambda_cc      :: Vector

 iter           :: Int64

 solved         :: Int64 #code de sortie

end

function RRelax(x              :: Vector;
                fx             :: Float64 = Inf,
                gx             :: Vector  = Float64[],
                feas           :: Vector  = Float64[],
                feas_cc        :: Vector  = Float64[],
                norm_feas      :: Float64 = Inf,
                dual_feas      :: Vector  = Float64[],
                norm_dual_feas :: Float64 = Inf,
                lambda         :: Vector  = Float64[],
                lambda_cc      :: Vector  = Float64[],
                iter           :: Int64   = 0,
                solved         :: Int64   = -1)


 return RRelax(x,fx,gx,feas,feas_cc,norm_feas,
               dual_feas,norm_dual_feas,lambda,lambda_cc,iter,solved)
end

function relax_start!(rrelax :: RRelax,
                      nlp    :: RlxMPCC,
                      xk     :: Vector;
                      fx     :: Float64 = Inf,
                      c      :: Vector  = Float64[])

 mod = nlp.mod
 r,s,t,tb = nlp.r, nlp.s, nlp.t, nlp.tb

 rrelax.fx = fx == Inf ? obj(mod,xk[1:mod.n]) : fx
 rrelax.feas, rrelax.feas_cc = cons(mod,r,s,t,tb,xk;cnl=c)
 rrelax.norm_feas = norm([rrelax.feas;rrelax.feas_cc], Inf)

 return rrelax
end

import MPCCmod.viol_cons, MPCCmod.viol_comp

function relax_update!(rrelax         :: RRelax,
                       mod            :: MPCC,
                       r              :: Float64,
                       s              :: Float64,
                       t              :: Float64,
                       tb             :: Float64,
                       xk             :: Vector,
                       rpen           :: RPen)

 c = viol_contrainte(mod, xk)
 cx,px = cons(mod,r,s,t,tb,xk;cnl=c[2*mod.ncc+1:length(c)])
 feas = vcat(cx,px)

 norm_feas = norm(feas, Inf)

 if norm(c,Inf) <= sqrt(eps(Float64))

  fx = rpen.fx 
  gx = rpen.gx
  dual_feas = rpen.dual_feas
  norm_dual_feas = norm(dual_feas, Inf)

 else

  fx = Inf #rpen.fx - penalty(c)
  gx = Float64[] #rpen.gx - grad_penalty(c)
  dual_feas = Float64[]
  norm_dual_feas = Inf

 end

 rrelax.iter = rrelax.iter + 1

 return relax_update!(rrelax, mod, r, s, t, tb, xk; 
                      fx = fx, gx = gx,
                      feas = cx, feas_cc = px,
                      norm_feas = norm_feas,
                      dual_feas = dual_feas,
                      norm_dual_feas = norm_dual_feas)
end

function relax_update!(rrelax         :: RRelax,
                       mod            :: MPCC,
                       r              :: Float64,
                       s              :: Float64,
                       t              :: Float64,
                       tb             :: Float64,
                       xk             :: Vector;
                       fx             :: Float64 = Inf,
                       gx             :: Vector  = Float64[],
                       feas           :: Vector  = Float64[],
                       feas_cc        :: Vector  = Float64[],
                       norm_feas      :: Float64 = Inf,
                       dual_feas      :: Vector  = Float64[],
                       norm_dual_feas :: Float64 = Inf,
                       lambda         :: Vector  = Float64[],
                       lambda_cc      :: Vector  = Float64[])

 rrelax.fx = fx == Inf ? obj(mod,xk[1:mod.n]) : fx
 rrelax.gx = gx == Float64[] ? rrelax.gx : gx

 rrelax.feas = feas == Float64[] ? rrelax.feas : feas
 rrelax.feas_cc = feas_cc == Float64[] ? rrelax.feas_cc : feas_cc
 rrelax.norm_feas = norm_feas == Inf ? rrelax.norm_feas : norm_feas

 rrelax.dual_feas = dual_feas == Float64[] ? rrelax.dual_feas : dual_feas
 rrelax.norm_dual_feas = norm_dual_feas == Inf ? rrelax.norm_dual_feas : norm_feas

 rrelax.lambda = lambda == Float64[] ? rrelax.lambda : lambda
 rrelax.lambda_cc = lambda_cc == Float64[] ? rrelax.lambda_cc : lambda_cc

 return rrelax
end

##################################################################################
#
# Additional functions
#
##################################################################################


############################################################################
#
# Feasibility of the relaxed problem
#
############################################################################
function cons(mod  :: MPCC,
              r    :: Float64,
              s    :: Float64,
              t    :: Float64,
              tb   :: Float64,
              x    :: Vector;
              cnl  :: Vector = Float64[])

 n = mod.n
 ncc = mod.ncc

 if ncc == 0

  c = viol_contrainte(mod,x) #les contraintes pénalisés

  xl = x

 else

  G = consG(mod,x[1:mod.n])
  H = consH(mod,x[1:mod.n])

  xl = length(x) == n ? vcat(x, G, H) : x

  c = vcat(G-xl[n+1:n+mod.ncc],H-xl[n+mod.ncc+1:n+2*mod.ncc],cnl)

 end

 slack = [max.(-xl[n+1:n+mod.ncc]+tb,0);
          max.(-xl[n+mod.ncc+1:n+2*mod.ncc]+tb,0)] #tb<= 0
 p = max.(phi(xl,mod.ncc,r,s,t),0)

 return c,[slack;p]
end


#end of module
end
