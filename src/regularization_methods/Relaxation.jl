###############################################################################
#Package of functions related to the Butterfly relaxation
#
#Citation:
#Dussault, J. P., Haddou, M., Kadrani, A., & Migot, T. (2017).
#How to Compute a M-stationary point of the MPCC.
###############################################################################

"""
psi(x) : relaxation function
"""
function psi(x :: Union{Float64,Vector},
             r :: Float64,
             s :: Float64,
             t :: Float64)

 return s .+ t.*theta(x .- s, r)
end

function invpsi(x :: Union{Float64,Vector},
                r :: Float64,
                s :: Float64,
                t :: Float64)

 #inverse not defined for t = 0.
 return t != 0 ? s .+ invtheta((x.-s)./t, r) : -Inf
end

function dpsi(x :: Union{Float64,Vector},
              r :: Float64,
              s :: Float64,
              t :: Float64)

 return t .* dtheta(x .- s, r)
end

function ddpsi(x :: Union{Float64,Vector},
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
function phi(yg :: Union{Float64,Vector},
             yh :: Union{Float64,Vector},
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
function dphi(yg :: Union{Float64,Vector},
              yh :: Union{Float64,Vector},
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
function ddphi(yg :: Union{Float64,Vector},
               yh :: Union{Float64,Vector},
               r  :: Float64,
               s  :: Float64,
               t  :: Float64)

 dgg = -2*dpsi(yg,r,s,t) - ddpsi(yg,r,s,t).*(yg - psi(yh,r,s,t))
 dhh = -2*dpsi(yh,r,s,t) - ddpsi(yh,r,s,t).*(yh - psi(yg,r,s,t))
 dgh = 1 .+ dpsi(yg,r,s,t).*dpsi(yh,r,s,t)

 return dgg, dgh, dhh
end

"""
alpha_max : see ThetaFct.alpha_theta_max

output : Array of size 1 x 2
"""
function alpha_max(x  :: Union{Float64,Vector},
                   dx :: Union{Float64,Vector},
                   y  :: Union{Float64,Vector},
                   dy :: Union{Float64,Vector},
                   r  :: Float64,
                   s  :: Float64,
                   t  :: Float64)

 return alpha_theta_max(x, dx, y, dy, r, s, t)
end
