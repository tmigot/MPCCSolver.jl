module RActifmod

import RPenmod.RPen

#résultat du problème pénalisé
type RActif

 x              :: Vector
 fx             :: Float64 #objective at x
 gx             :: Vector #gradient at x

 feas           :: Vector #violation of the constraints
 feas_cc        :: Vector #violation of the relaxed constraints
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

function RActif(x              :: Vector;
                fx             :: Float64 = Inf,
                gx             :: Vector  = Float64[],
                feas           :: Vector  = Float64[],
                feas_cc        :: Vector  = Float64[],
                norm_feas      :: Float64 = Inf,
                dual_feas      :: Vector  = Float64[],
                norm_dual_feas :: Float64 = Inf,
                lambda         :: Vector  = Float64[],
                wnew           :: Array = zeros(Bool,0,0),
                step           :: Float64 = 1.0,
                sub_pb_solved  :: Bool = true,
                iter           :: Int64 = 0,
                solved         :: Int64   = -1)


 return RActif(x,fx,gx,feas,feas_cc,norm_feas,
               dual_feas,norm_dual_feas,lambda,
               wnew,step,sub_pb_solved,iter,solved)
end

function actif_start!(ractif  :: RActif,
                      xk      :: Vector,
                      rpen    :: RPen,
                      ncc     :: Int64)

  n = length(xk) - 2*ncc

  ractif.feas      = rpen.feas[1:2*n]
  ractif.feas_cc   = rpen.feas[2*n+1:2*n+3*ncc]
  ractif.norm_feas = norm(rpen.feas, Inf)
  ractif.fx        = rpen.fx
  ractif.gx        = rpen.gx
  ractif.step      = rpen.step
  ractif.wnew      = rpen.wnew

 return ractif
end

function update!(ractif :: RActif,
                 xk             :: Vector;
                 fx             :: Float64 = Inf,
                 gx             :: Vector  = Float64[],
                 feas           :: Vector  = Float64[],
                 norm_feas      :: Float64 = Inf,
                 dual_feas      :: Vector  = Float64[],
                 norm_dual_feas :: Float64 = Inf,
                 lambda         :: Vector  = Float64[])

 ractif.fx = fx == Inf ? obj(mod,xk) : fx
 ractif.gx = gx == Float64[] ? ractif.gx : gx

 ractif.feas = feas == Float64[] ? ractif.feas : feas

 ractif.dual_feas = dual_feas == Float64[] ? ractif.dual_feas : dual_feas
 ractif.norm_dual_feas = norm_dual_feas == Inf ? ractif.norm_dual_feas : norm_feas

 ractif.lambda = lambda == Float64[] ? ractif.lambda : lambda

 return ractif
end

##################################################################################
#
# Additional functions
#
##################################################################################


#end of module
end
