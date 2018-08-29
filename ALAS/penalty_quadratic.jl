##########################################################################
#
# P(hx,bx,gx) = 1/2 * (||h(x)||^2+||b(x)||^2+||g(x)||^2)
#
##########################################################################
function quadratic(hx  :: Vector,
                   bx  :: Vector,
                   gx  :: Vector,
                   rho :: Vector,
                   u   :: Vector)

 lh, lb, lg = length(hx), length(bx), length(gx)
 rhoh, rhob, rhog = rho[1:lh], rho[lh+1:lh+lb], rho[lh+lb+1:lh+lb+lg]

 return 0.5*(dot(rhoh.*hx,hx)+dot(rhog.*gx,gx)+dot(rhob.*bx,bx))
end

##########################################################################
#
# \nabla P(hx,bx,gx)
#
##########################################################################
function quadratic(hx    :: Vector,
                   bx    :: Vector,
                   gx    :: Vector,
                   grad  :: Vector,
                   rho   :: Vector,
                   u     :: Vector)

 lh, lb, lg = length(hx), length(bx), length(gx)
 rhoh, rhob, rhog = rho[1:lh], rho[lh+1:lh+lb], rho[lh+lb+1:lh+lb+lg]

 grad = vcat(vcat(hx,bx,gx).*rho,hx.*rhoh)

 return grad
end

##########################################################################
#
# \nabla P(hx,bx,gx), \nabla^2 P(hx,bx,gx)
#
##########################################################################
function quadratic(hx    :: Vector,
                   bx    :: Vector,
                   gx    :: Vector,
                   grad  :: Vector,
                   Hx    :: Any, #matrix type
                   rho   :: Vector,
                   u     :: Vector)

 lh, lb, lg = length(hx), length(bx), length(gx)
 rhoh, rhob, rhog = rho[1:lh], rho[lh+1:lh+lb], rho[lh+lb+1:lh+lb+lg]

 grad = vcat(vcat(hx,bx,gx).*rho,hx.*rhoh)
 Hx = diagm(rho)

 return grad, Hx
end
