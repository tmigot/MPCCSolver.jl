#Tests des packages
include("../ALAS/Relaxationmod.jl")

using Relaxationmod


r=1.0;s=1.0;t=1.0;
Relaxationmod_success=true

if norm(Relaxationmod.psi(s,r,s,t)-s)>precision
 x = linspace(-0.5,10,1000); y = Relaxationmod.psi(x,r,s,t)
 PyPlot.plot(x, y, color="red", linewidth=2.0, linestyle="-")
 x = linspace(-0.5,10,1000); y = Relaxationmod.psi(x,r,s,t)
 PyPlot.plot(y, x, color="blue", linewidth=2.0, linestyle="-")
end

#AlphaThetaMax(x::Float64,dx::Float64,y::Float64,dy::Float64,r::Float64,s::Float64,t::Float64)
#AlphaThetaMax : résoud l'équation en alpha où psi(x,r,s,t)=s+t*theta(x,r)
r=1.0;s=1.0;t=0.0
x=1.0;y=2.0;dx=1.0;dy=-1.0;
println(Relaxationmod.AlphaThetaMax(x,dx,y,dy,r,s,t))
#doit rendre (0,1)
x=0.0;y=2.0;dx=1.0;dy=-2.0;
println(Relaxationmod.AlphaThetaMax(x,dx,y,dy,r,s,t))
#solution : (1,0.5)

r=1.0;s=1.0;t=1.0
x=1.0;y=2.0;dx=1.0;dy=-1.0;
println(Relaxationmod.AlphaThetaMax(x,dx,y,dy,r,s,t))
#doit rendre [0.381966,0.618034]
x=0.0;y=2.0;dx=1.0;dy=-2.0;
println(Relaxationmod.AlphaThetaMax(x,dx,y,dy,r,s,t))
#solution : [0.640388,0.697224]

#Bilan : donner une sortie bilan
if Relaxationmod_success==true
 println("Relaxationmod.jl passes the test !")
else
 println("Relaxationmod.jl contains some error")
end
