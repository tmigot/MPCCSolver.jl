module RPenmod

import PenMPCCmod.PenMPCC
import PenMPCCmod.jac, PenMPCCmod.cons
import PenMPCCmod.obj, PenMPCCmod.grad

type RPen #résultat du problème pénalisé

 x              :: Vector
 fx             :: Float64 #objective at x
 gx             :: Vector #gradient at x

 feas           :: Vector #violation of the constraints
 norm_feas      :: Float64 # norm of feas

 dual_feas      :: Vector
 norm_dual_feas :: Float64

 lambda         :: Vector

 wnew           :: Array #last added constraints
 step           :: Float64 #last step in the sub-problem
 sub_pb_solved  :: Bool
 iter           :: Int64

 solved         :: Int64 #code de sortie

end

function RPen(x              :: Vector;
              fx             :: Float64 = Inf,
              gx             :: Vector  = Float64[],
              feas           :: Vector  = Float64[],
              norm_feas      :: Float64 = Inf,
              dual_feas      :: Vector  = Float64[],
              norm_dual_feas :: Float64 = Inf,
              lambda         :: Vector  = Float64[],
              wnew           :: Array   = zeros(Bool,0,0),
              step           :: Float64 = 1.0,
              sub_pb_solved  :: Bool    = true,
              iter           :: Int64   = 0,
              solved         :: Int64   = -1)


 return RPen(x,fx,gx,feas,norm_feas,
               dual_feas,norm_dual_feas,lambda,
               wnew,step,sub_pb_solved,iter,solved)
end

function pen_start!(pen            :: PenMPCC,
                    rpen           :: RPen,
                    xk             :: Vector;
                    fx             :: Float64 = Inf,
                    gx             :: Vector  = Float64[],
                    feas           :: Vector  = Float64[],
                    norm_feas      :: Float64 = Inf,
                    dual_feas      :: Vector  = Float64[],
                    norm_dual_feas :: Float64 = Inf,
                    lambda         :: Vector  = zeros(2*pen.n+3*pen.ncc))

 rpen.fx = rpen.fx == Inf ? obj(pen,xk) : rpen.fx
 rpen.gx = rpen.gx == Float64[] ? grad(pen,xk) : rpen.gx
 #rpen.lambda = norm(lambda,Inf) != 0.0 ? lambda : rpen.lambda #vide ?
 rpen.lambda = lambda

 Jl = jac(pen, xk, rpen.lambda)
 rpen.dual_feas = rpen.dual_feas == Float64[] ? rpen.gx + Jl : rpen.dual_feas

 rpen.feas = rpen.feas == Float64[] ? cons(pen, xk) : rpen.feas
 rpen.norm_feas = norm(rpen.feas, Inf)

 return RPen
end

function pen_update!(pen            :: PenMPCC,
                     rpen           :: RPen,
                     xk             :: Vector,
                     sub_pb_solved  :: Bool;
                     fx             :: Float64 = Inf,
                     gx             :: Vector  = Float64[],
                     feas           :: Vector  = Float64[],
                     norm_feas      :: Float64 = Inf,
                     dual_feas      :: Vector  = Float64[],
                     norm_dual_feas :: Float64 = Inf,
                     lambda         :: Vector  = zeros(2*pen.n+3*pen.ncc),
                     step           :: Float64 = NaN,
                     wnew           :: Array   = Bool[])

 rpen.sub_pb_solved = sub_pb_solved

 rpen.fx = fx == Inf ? rpen.fx : fx
 rpen.gx = gx == Float64[] ? rpen.gx : gx

 rpen.feas = feas == Float64[] ? rpen.feas : feas

 rpen.dual_feas = dual_feas == Float64[] ? rpen.dual_feas : dual_feas
 rpen.norm_dual_feas = norm_dual_feas == Inf ? rpen.norm_dual_feas : norm_feas

 rpen.lambda = lambda == Float64[] ? rpen.lambda : lambda

 rpen.step = isnan(step) ? rpen.step : step  

 rpen.wnew = wnew == Bool[] ? rpen.wnew : wnew

 rpen.iter+=1
 return rpen
end

function pen_rho_update!(pen  :: PenMPCC,
                         rpen :: RPen,
                         xk   :: Vector)

  lambda = rpen.lambda

  rpen.dual_feas = rpen.gx + jac(pen, xk, rpen.lambda)
  rpen.feas = cons(pen, xk)

 return rpen
end

##################################################################################
#
# Additional functions
#
##################################################################################


#end of module
end
