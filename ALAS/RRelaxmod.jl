module RRelaxmod

import MPCCmod.MPCC, MPCCmod.obj, MPCCmod.grad
import MPCCmod.consG, MPCCmod.consH
import MPCCmod.viol_contrainte

import Relaxation.phi

type RRelax

 x :: Vector
 fx :: Float64 #objective at x
 gx :: Vector #gradient at x

 feas :: Vector #violation of the constraints
 feas_cc :: Vector #violation of the relaxed constraints
 norm_feas :: Float64 # norm of feas

 dual_feas :: Vector
 norm_dual_feas :: Float64

 lambda :: Vector
 lambda_cc :: Vector

 solved :: Int64 #code de sortie

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
                solved         :: Int64   = -1)


 return RRelax(x,fx,gx,feas,feas_cc,norm_feas,
               dual_feas,norm_dual_feas,lambda,lambda_cc,solved)
end

function relax_start!(rrelax :: RRelax,
                      mod    :: MPCC,
                      r      :: Float64,
                      s      :: Float64,
                      t      :: Float64,
                      tb     :: Float64,
                      xk     :: Vector)

 rrelax.fx = rrelax.fx == Inf ? obj(mod,xk[1:mod.n]) : rrelax.fx
 rrelax.feas, rrelax.feas_cc = cons(mod,r,s,t,tb,xk;cnl=rrelax.feas)
 rrelax.norm_feas = norm([rrelax.feas;rrelax.feas_cc], Inf)

 return rrelax
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
                       norm_feas      :: Float64 = Inf,
                       dual_feas      :: Vector  = Float64[],
                       norm_dual_feas :: Float64 = Inf,
                       lambda         :: Vector  = Float64[],
                       lambda_cc      :: Vector  = Float64[])

 rrelax.fx = fx == Inf ? obj(mod,xk) : fx
 rrelax.gx = gx == Float64[] ? rrelax.gx : gx

 rrelax.feas = feas == Float64[] ? rrelax.feas : feas
 #rrelax.norm_feas = norm_feas == Inf ? viol_contrainte_norm(mod,xk,tnorm=Inf) : norm_feas

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

 if length(cnl) == mod.mp.meta.ncon+2*n

  c = viol_contrainte(mod,x) #les contraintes pénalisés

 else

  G = consG(mod,x[1:mod.n])
  H = consH(mod,x[1:mod.n])

  c = vcat(G-x[n+1:n+mod.nb_comp],H-x[n+mod.nb_comp+1:n+2*mod.nb_comp],cnl)

 end

 slack = [max.(-x[n+1:n+mod.nb_comp]+tb,0);
          max.(-x[n+mod.nb_comp+1:n+2*mod.nb_comp]+tb,0)] #tb<= 0
 p = max.(phi(x,mod.nb_comp,r,s,t),0)

 return c,[slack;p]
end


#end of module
end
