module RUncstrndmod

#résultat du problème pénalisé
type RUncstrnd

 x              :: Vector #x at the previous iteration

 fx             :: Float64 #objective at x
 gx             :: Vector #gradient at x
 fxp            :: Float64 #objective at x
 gxp            :: Vector #gradient at x

 ∇f             :: Vector #gradient projected at x
 ∇fp            :: Vector #gradient projected at x

 cons           :: Vector #violation of the constraints

 step           :: Float64 #last step in the sub-problem
 d              :: Vector #gradient at x

 iter           :: Int64

 solved         :: Int64 #code de sortie

 function RUncstrnd(x              :: Vector;
                    fx             :: Float64 = Inf,
                    gx             :: Vector  = Float64[],
                    fxp            :: Float64 = Inf,
                    gxp            :: Vector  = Float64[],
                                   ∇f             :: Vector  = Float64[],
                                   ∇fp            :: Vector  = Float64[],
                    cons           :: Vector  = Float64[],
                    step           :: Float64 = 1.0,
                    d              :: Vector  = Float64[],
                    iter           :: Int64 = 0,
                    solved         :: Int64 = -1)


  return new(x,fx,gx,fxp,gxp,∇f,∇fp,cons,step,d,iter,solved)
 end
end

function runc_start!(runc :: RUncstrnd;
                     fx     :: Float64 = Inf,
                     gx     :: Vector  = Float64[],
                     fxp    :: Float64 = Inf,
                     gxp    :: Vector  = Float64[],
                                     ∇f     :: Vector  = Float64[],
                                     ∇fp    :: Vector  = Float64[],
                     cons   :: Vector  = Float64[],
                     step   :: Float64 = 1.0,
                     d      :: Vector  = Float64[],
                     iter   :: Int64 = 0,
                     solved :: Int64 = -1)

 runc.fx = fx == Inf ? runc.fx : fx
 runc.gx = gx == Float64[] ? runc.gx : gx

 runc.fxp = fxp == Inf ? runc.fxp : fxp
 runc.gxp = gxp == Float64[] ? runc.gxp : gxp

 runc.∇f  = ∇f  == Float64[] ? runc.∇f  : ∇f
 runc.∇fp = ∇fp == Float64[] ? runc.∇fp : ∇fp

 runc.cons = cons == Float64[] ? runc.cons : cons

 runc.d = d == Float64[] ? runc.d : d

 return runc
end

function runc_update!(runc   :: RUncstrnd,
                      xk     :: Vector;
                      fx     :: Float64 = Inf,
                      gx     :: Vector  = Float64[],
                      fxp    :: Float64 = Inf,
                      gxp    :: Vector  = Float64[],
                                       ∇f     :: Vector  = Float64[],
                                       ∇fp    :: Vector  = Float64[],
                      cons   :: Vector  = Float64[],
                      step   :: Float64 = 1.0,
                      d      :: Vector  = Float64[],
                      iter   :: Int64 = -1,
                      solved :: Int64 = -1)

 runc.fx = fx == Inf ? runc.fx : fx
 runc.gx = gx == Float64[] ? runc.gx : gx

 runc.fxp = fxp == Inf ? runc.fxp : fxp
 runc.gxp = gxp == Float64[] ? runc.gxp : gxp

 runc.∇f  = ∇f  == Float64[] ? runc.∇f : ∇f
 runc.∇fp = ∇fp == Float64[] ? runc.∇fp : ∇fp

 runc.cons = cons == Float64[] ? runc.cons : cons

 runc.d = d == Float64[] ? runc.d : d

 runc.step = step == 1.0 ? runc.step : step

 runc.iter = iter == -1 ? runc.iter+1 : iter

 return runc
end

#end of module
end
