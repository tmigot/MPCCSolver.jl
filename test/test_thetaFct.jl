r=0.1

#theta(x::FloatOrVector,r::Float64)
#theta's vanishe in 0, are bounded by 1, increasing, concave
# positive for positive values, and negative for negative values.
@test MPCCSolver.theta(0.0, r) .== 0.0
@test MPCCSolver.theta(1.0+1e6, r) .<= 1
@test MPCCSolver.theta(-1e-6, r) .< 0.0
@test MPCCSolver.theta(1.0, r) .>= MPCCSolver.theta(0.5, r)

#dtheta(x::FloatOrVector,r::Float64)
#la dérivée des fonctions theta doit être positive
@test MPCCSolver.dtheta(1.0+1e6, r) >= 0.0
@test MPCCSolver.dtheta(-0.5, r)    >= 0.0

#ddtheta(x::FloatOrVector,r::Float64)
#la dérivée seconde des fonctions theta doit être négative
@test MPCCSolver.ddtheta(1.0+1e6, r) <= 0.0
@test MPCCSolver.ddtheta(-0.5, r)    <= 0.0

#invtheta(x::Vector,r::Float64)
@test abs(MPCCSolver.invtheta(MPCCSolver.theta(1.,r),r)-1.)      <= eps(Float64)*10.
@test abs(MPCCSolver.invtheta(MPCCSolver.theta(-1e-6,r),r)+1e-6) <= eps(Float64)*10.

#_poly_degree_2_positif
#Compute the smallest positive root of a second order polynomial
#or a negative/complex value if no positive real root.
#If no solution or infinite nb of solutions, return Inf.
@test abs(MPCCSolver._poly_degree_2_positif(0.5,-1.,0.25) - 0.2928932188134524) <= eps(Float64)
@test abs(MPCCSolver._poly_degree_2_positif(0.,1.,-0.5) - 0.5) <= eps(Float64)
@test !isfinite(MPCCSolver._poly_degree_2_positif(0.,0.,-0.))
@test !isfinite(MPCCSolver._poly_degree_2_positif(1.,1.,1.))

#alpha_theta_max(x::Float64,dx::Float64,y::Float64,dy::Float64,r::Float64,s::Float64,t::Float64)
#alpha_theta_max : résoud l'équation en alpha où psi(x,r,s,t)=s+t*theta(x,r)
r=1.0;s=1.0;t=0.0 #résoud l'éq.

x=1.0;y=2.0;dx=1.0;dy=-1.0;
#doit rendre (0,1)
@test isequal(MPCCSolver.alpha_theta_max(x,dx,y,dy,r,s,t), [0.0,1.0])

x=0.0;y=2.0;dx=1.0;dy=-2.0;
#solution : (1,0.5)

@test isequal(MPCCSolver.alpha_theta_max(x,dx,y,dy,r,s,t), [1.0,0.5])

r=1.0;s=1.0;t=1.0
#on tape la contrainte :
x=1.0;y=2.0;dx=1.0;dy=0.0;

@test minimum(MPCCSolver.alpha_theta_max(x,dx,y,dy,r,s,t))>=0.0
@test maximum(MPCCSolver.alpha_theta_max(x,dx,y,dy,r,s,t))==Inf

x=2.0;y=1.0;dx=-1.0;dy=2.0;

@test minimum(MPCCSolver.alpha_theta_max(x,dx,y,dy,r,s,t))>=0.0
@test maximum(MPCCSolver.alpha_theta_max(x,dx,y,dy,r,s,t))<Inf

#on est sous (s,s) et donc on ne tape pas la contrainte virtuelle
x=0.0;y=0.0;dx=0.0;dy=-2.0;

@test isequal(maximum(MPCCSolver.alpha_theta_max(x,dx,y,dy,r,s,t)),Inf)
