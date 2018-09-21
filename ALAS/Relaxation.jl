"""
Package of functions related to the Butterfly relaxation

Citation:
Dussault, J. P., Haddou, M., Kadrani, A., & Migot, T. (2017). 
How to Compute a M-stationary point of the MPCC.
"""
module Relaxation

const FloatOrVector = Union{Float64,Vector}

import ThetaFct.theta
import ThetaFct.invtheta
import ThetaFct.dtheta, ThetaFct.ddtheta
import ThetaFct.alpha_theta_max

"""
psi(x) : relaxation function
"""
function psi(x :: FloatOrVector,
             r :: Float64, 
             s :: Float64,
             t :: Float64)

 return s .+ t.*theta(x .- s, r)
end

function invpsi(x :: FloatOrVector,
                r :: Float64, 
                s :: Float64,
                t :: Float64)

 #inverse not defined for t = 0.
 return t != 0 ? s .+ invtheta((x.-s)./t, r) : -Inf
end

function dpsi(x :: FloatOrVector,
              r :: Float64, 
              s :: Float64,
              t :: Float64)

 return t .* dtheta(x .- s, r)
end

function ddpsi(x :: FloatOrVector,
               r :: Float64, 
               s :: Float64,
               t :: Float64)

 return t .* ddtheta(x .- s,r)
end

"""
phi(x,r,s,t) : évalue la fonction de relaxation
de la contrainte de complémentarité en (yg,yh) in (ncc,ncc)

output : Array of size 1 x ncc
"""
function phi(yg :: FloatOrVector,
             yh :: FloatOrVector,
             r  :: Float64,
             s  :: Float64,
             t  :: Float64)

 return (yg - psi(yh, r, s, t)) .* (yh - psi(yg, r, s, t))
end
"""
phi(x,r,s,t) : évalue la fonction de relaxation
 de la contrainte de complémentarité en x in n+2ncc

output : Array of size 1 x ncc
"""
function phi(x   :: Vector,
             ncc :: Int64,
             r   :: Float64,
             s   :: Float64,
             t   :: Float64)

 n = length(x)-2*ncc

 yg = x[n+1:n+ncc]
 yh = x[n+ncc+1:n+2*ncc]

 return phi(yg, yh, r, s, t)
end

"""
dphi(x,r,s,t) : évalue la fonction de relaxation de la contrainte de complémentarité en (yg,yh) in (ncc,ncc)

output : Array of size 1 x 2ncc
"""
function dphi(yg :: FloatOrVector,
              yh :: FloatOrVector,
              r  :: Float64,
              s  :: Float64,
              t  :: Float64)

 dg = (yh - psi(yg,r,s,t)) - dpsi(yg,r,s,t).*(yg - psi(yh,r,s,t))
 dh = (yg - psi(yh,r,s,t)) - dpsi(yh,r,s,t).*(yh - psi(yg,r,s,t))

 return [dg;dh]
end

"""
dphi(yg,yh,r,s,t) : évalue la fonction de relaxation de la contrainte de complémentarité en x in n+2ncc

output : Array of size 1 x (n +2ncc)
"""
function dphi(x   :: Vector,
              ncc :: Int64,
              r   :: Float64,
              s   :: Float64,
              t   :: Float64)

 n = length(x)-2*ncc

 yg = x[n+1:n+ncc]
 yh = x[n+ncc+1:n+2*ncc]

 return [zeros(n);dphi(yg,yh,r,s,t)]
end

"""
ddphi(x,r,s,t) : évalue la fonction de relaxation de la contrainte de complémentarité en (yg,yh) in (ncc,ncc)

output : Array of size 2ncc x 2ncc
"""
function ddphi(yg :: FloatOrVector,
               yh :: FloatOrVector,
               r  :: Float64,
               s  :: Float64,
               t  :: Float64)

 dgg = -2*dpsi(yg,r,s,t) - ddpsi(yg,r,s,t).*(yg - psi(yh,r,s,t))
 dhh = -2*dpsi(yh,r,s,t) - ddpsi(yh,r,s,t).*(yh - psi(yg,r,s,t))
 dgh = 1 + dpsi(yg,r,s,t).*dpsi(yh,r,s,t)

 return dgg, dgh, dhh
end

"""
alpha_max : see ThetaFct.alpha_theta_max

output : Array of size 1 x 2
"""
function alpha_max(x  :: FloatOrVector,
                   dx :: FloatOrVector,
                   y  :: FloatOrVector,
                   dy :: FloatOrVector,
                   r  :: Float64,
                   s  :: Float64,
                   t  :: Float64)

 return alpha_theta_max(x, dx, y, dy, r, s, t)
end

#end of module
end
