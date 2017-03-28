println("On teste le package Relaxation")

using Relaxation


r=1.0;s=1.0;t=1.0;
Relaxation_success=true

# psi(x::Any,r::Float64,s::Float64,t::Float64)
if norm(Relaxation.psi(s,r,s,t)-s)>precision
 x = linspace(-0.5,10,1000); y = Relaxation.psi(x,r,s,t)
 PyPlot.plot(x, y, color="red", linewidth=2.0, linestyle="-")
 x = linspace(-0.5,10,1000); y = Relaxation.psi(x,r,s,t)
 PyPlot.plot(y, x, color="blue", linewidth=2.0, linestyle="-")
end

#invpsi(x::Any,r::Float64,s::Float64,t::Float64)
if norm(Relaxation.invpsi(s,r,s,t)-s)>precision
 x = linspace(-0.5,10,1000); y = Relaxation.psi(x,r,s,t)
 PyPlot.plot(x, y, color="red", linewidth=2.0, linestyle="-")
 x = linspace(-0.5,10,1000); y = Relaxation.psi(x,r,s,t)
 PyPlot.plot(y, x, color="blue", linewidth=2.0, linestyle="-")
end
#dpsi(x::Any,r::Float64,s::Float64,t::Float64)
#phi(yg::Any,yh::Any,r::Float64,s::Float64,t::Float64)
#phi(x::Any,nb_comp::Int64,r::Float64,s::Float64,t::Float64)
#dphi(yg::Any,yh::Any,r::Float64,s::Float64,t::Float64)
#dphi(x::Any,nb_comp::Int64,r::Float64,s::Float64,t::Float64)

#AlphaThetaMax(x::Float64,dx::Float64,y::Float64,dy::Float64,r::Float64,s::Float64,t::Float64)
#AlphaThetaMax : résoud l'équation en alpha où psi(x,r,s,t)=s+t*theta(x,r)
r=1.0;s=1.0;t=0.0 #résoud l'éq.

x=1.0;y=2.0;dx=1.0;dy=-1.0;
#doit rendre (0,1)
if !isequal(Relaxation.AlphaThetaMax(x,dx,y,dy,r,s,t),[0.0,1.0])
 println(Relaxation.AlphaThetaMax(x,dx,y,dy,r,s,t))
 Thetafunc_success=false
end

x=0.0;y=2.0;dx=1.0;dy=-2.0;
#solution : (1,0.5)
if !isequal(Relaxation.AlphaThetaMax(x,dx,y,dy,r,s,t),[1.0,0.5])
 println(Relaxation.AlphaThetaMax(x,dx,y,dy,r,s,t))
 Relaxation_success=false
end

r=1.0;s=1.0;t=1.0
#on tape la contrainte :
x=1.0;y=2.0;dx=-1.0;dy=0.0;
if !(minimum(Relaxation.AlphaThetaMax(x,dx,y,dy,r,s,t))>=0.0) || !(maximum(Relaxation.AlphaThetaMax(x,dx,y,dy,r,s,t))<Inf)
 println(Relaxation.AlphaThetaMax(x,dx,y,dy,r,s,t))
 Relaxation_success=false
end
x=0.0;y=2.0;dx=1.0;dy=-2.0;
if !(minimum(Relaxation.AlphaThetaMax(x,dx,y,dy,r,s,t))>=0.0) || !(maximum(Relaxation.AlphaThetaMax(x,dx,y,dy,r,s,t))<Inf)
 println(Relaxation.AlphaThetaMax(x,dx,y,dy,r,s,t))
 Relaxation_success=false
end

#on est sous (s,s) et donc on ne tape pas la contrainte virtuelle
x=0.0;y=0.0;dx=0.0;dy=-2.0;
if !isequal(maximum(Relaxation.AlphaThetaMax(x,dx,y,dy,r,s,t)),Inf)
 println(Relaxation.AlphaThetaMax(x,dx,y,dy,r,s,t))
 Relaxation_success=false
end

#Bilan : donner une sortie bilan
if Relaxation_success==true
 println("Relaxationmod.jl passes the test !")
else
 println("Relaxationmod.jl contains some error")
end
