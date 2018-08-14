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
                    step           :: Float64 = 1.0,
                    d              :: Vector  = Float64[],
                    iter           :: Int64 = 0,
                    solved         :: Int64 = -1)


  return new(x,fx,gx,fxp,gxp,∇f,∇fp,step,d,iter,solved)
 end
end

function runc_start!(RUncstrnd :: RUncstrnd;
                 fx             :: Float64 = Inf,
                 gx             :: Vector  = Float64[],
                 fxp            :: Float64 = Inf,
                 gxp            :: Vector  = Float64[],
                              ∇f             :: Vector  = Float64[],
                               ∇fp            :: Vector  = Float64[],
                 step           :: Float64 = 1.0,
                 d              :: Vector  = Float64[],
                 iter           :: Int64 = 0,
                 solved         :: Int64 = -1)

 RUncstrnd.fx = fx == Inf ? RUncstrnd.fx : fx
 RUncstrnd.gx = gx == Float64[] ? RUncstrnd.gx : gx

 RUncstrnd.fxp = fxp == Inf ? RUncstrnd.fxp : fxp
 RUncstrnd.gxp = gxp == Float64[] ? RUncstrnd.gxp : gxp

 RUncstrnd.∇f  = ∇f  == Float64[] ? RUncstrnd.∇f  : ∇f
 RUncstrnd.∇fp = ∇fp == Float64[] ? RUncstrnd.∇fp : ∇fp

 RUncstrnd.d = d == Float64[] ? RUncstrnd.d : d

 RUncstrnd.step += 1 #WTF !!!

 return RUncstrnd
end

function runc_update!(RUncstrnd :: RUncstrnd,
                      xk             :: Vector;
                      fx             :: Float64 = Inf,
                      gx             :: Vector  = Float64[],
                      fxp            :: Float64 = Inf,
                      gxp            :: Vector  = Float64[],
                                       ∇f             :: Vector  = Float64[],
                                       ∇fp            :: Vector  = Float64[],
                      step           :: Float64 = 1.0,
                      d              :: Vector  = Float64[],
                      iter           :: Int64 = -1,
                      solved         :: Int64 = -1)

 RUncstrnd.fx = fx == Inf ? RUncstrnd.fx : fx
 RUncstrnd.gx = gx == Float64[] ? RUncstrnd.gx : gx

 RUncstrnd.fxp = fxp == Inf ? RUncstrnd.fxp : fxp
 RUncstrnd.gxp = gxp == Float64[] ? RUncstrnd.gxp : gxp

 RUncstrnd.∇f = ∇f == Float64[] ? RUncstrnd.∇f : ∇f
 RUncstrnd.∇fp = ∇fp == Float64[] ? RUncstrnd.∇fp : ∇fp

 RUncstrnd.d = d == Float64[] ? RUncstrnd.d : d

 RUncstrnd.step = step == 1.0 ? RUncstrnd.step : step

 RUncstrnd.iter = iter == -1 ? RUncstrnd.iter+1 : iter

 return RUncstrnd
end

#end of module
end
