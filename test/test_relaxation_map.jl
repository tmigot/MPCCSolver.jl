r=1.0;s=1.0;t=1.0;

Relaxation_success = true

# psi(x::Any,r::Float64,s::Float64,t::Float64)
@test norm(MPCCSolver.psi(s,r,s,t)-s) <= eps(Float64)*10
# x = linspace(-0.5,10,1000); y = MPCCSolver.psi(x,r,s,t)
# PyPlot.plot(x, y, color="red", linewidth=2.0, linestyle="-")
# x = linspace(-0.5,10,1000); y = MPCCSolver.psi(x,r,s,t)
# PyPlot.plot(y, x, color="blue", linewidth=2.0, linestyle="-")


#invpsi(x::Any,r::Float64,s::Float64,t::Float64)
@test norm(MPCCSolver.invpsi(s,r,s,t)-s) <= eps(Float64)*10

# x = linspace(-0.5,10,1000); y = MPCCSolver.psi(x,r,s,t)
# PyPlot.plot(x, y, color="red", linewidth=2.0, linestyle="-")
# x = linspace(-0.5,10,1000); y = MPCCSolver.psi(x,r,s,t)
# PyPlot.plot(y, x, color="blue", linewidth=2.0, linestyle="-")

#dpsi(x::Any,r::Float64,s::Float64,t::Float64)
#phi(yg::Any,yh::Any,r::Float64,s::Float64,t::Float64)
#phi(x::Any,nb_comp::Int64,r::Float64,s::Float64,t::Float64)
#dphi(yg::Any,yh::Any,r::Float64,s::Float64,t::Float64)
#dphi(x::Any,nb_comp::Int64,r::Float64,s::Float64,t::Float64)

#alpha_max(x::Float64,dx::Float64,y::Float64,dy::Float64,r::Float64,s::Float64,t::Float64)
#alpha_max : résoud l'équation en alpha où psi(x,r,s,t)=s+t*theta(x,r)
r=1.0;s=1.0;t=0.0 #résoud l'éq.

x=1.0;y=2.0;dx=1.0;dy=-1.0;
#doit rendre (0,1)
@test isequal(MPCCSolver.alpha_max(x,dx,y,dy,r,s,t),[0.0,1.0])

x=0.0;y=2.0;dx=1.0;dy=-2.0;
#solution : (1,0.5)
@test isequal(MPCCSolver.alpha_max(x,dx,y,dy,r,s,t),[1.0,0.5])

r=1.0;s=1.0;t=1.0
#on tape la contrainte :
x=1.0;y=2.0;dx=1.0;dy=0.0;

@test minimum(MPCCSolver.alpha_max(x,dx,y,dy,r,s,t))>=0.0
@test maximum(MPCCSolver.alpha_max(x,dx,y,dy,r,s,t))==Inf

x=2.0;y=1.0;dx=-1.0;dy=2.0;

@test minimum(MPCCSolver.alpha_max(x,dx,y,dy,r,s,t)) >= 0.0
@test maximum(MPCCSolver.alpha_max(x,dx,y,dy,r,s,t)) <  Inf

#on est sous (s,s) et donc on ne tape pas la contrainte virtuelle
x=0.0;y=0.0;dx=0.0;dy=-2.0;

@test isequal(maximum(MPCCSolver.alpha_max(x,dx,y,dy,r,s,t)),Inf)

#Conclusion:
@test Relaxation_success
