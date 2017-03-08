#Tests des packages
include("../ALAS/Thetamod.jl")

using Thetamod

Thetamod_success=true
r=0.1

#theta(x::Vector,r::Float64)
#Les fonctions thetas doivent etre nulle en 0, <=1 pour x>=0 et <0 pour x<0
if Thetamod.theta(0.0,r)!=0 && Thetamod.theta(1.0+1e6,r)>=1 && Thetamod(-1e-6,r)>=0
 println("Error in the definition of theta")
 plot(x->Thetamod.theta(x,r),-0.1,2.0)
 #draw(SVG("output.svg",6inch,3inch),plot(x->Thetamod.theta(x,r),-0.1,2.0))
 Thetamod_success=false
end

#dtheta(x::Vector,r::Float64)
#la dérivée des fonctions theta doit être croissante
if Thetamod.theta(1.0+1e6,r)>=0 && (Thetamod.theta(1.0,r)-Thetamod.theta(2.0,r)>0)
 println("Error in the definition of dtheta")
 plot(x->Thetamod.dtheta(x,r),-0.1,2.0)
 Thetamod_success=false
end

#invtheta(x::Vector,r::Float64)
#Tester aussi les cas interdits ici pas défini pour x>1 !!!
if norm(Thetamod.invtheta(Thetamod.theta(1e-6,r),r)-1e-6)>precision || norm(Thetamod.invtheta(Thetamod.theta(-1e-6,r),r)+1e-6)>precision
 println("Error in the definition of invtheta")
 println(norm(Thetamod.invtheta(Thetamod.theta(1e-6,r),r)-1e-6))
 println(norm(Thetamod.invtheta(Thetamod.theta(-1e-6,r),r)+1e-6))
 plot(x->Thetamod.invtheta(Thetamod.theta(x,r),r),-0.1,1.0)
 Thetamod_success=false
end

#AlphaThetaMax(x::Float64,dx::Float64,y::Float64,dy::Float64,r::Float64,s::Float64,t::Float64)
#AlphaThetaMax : résoud l'équation en alpha où psi(x,r,s,t)=s+t*theta(x,r)
r=1.0;s=1.0;t=0.0
x=1.0;y=2.0;dx=1.0;dy=-1.0;
println(Thetamod.AlphaThetaMax(x,dx,y,dy,r,s,t))
#doit rendre (0,1)
x=0.0;y=2.0;dx=1.0;dy=-2.0;
println(Thetamod.AlphaThetaMax(x,dx,y,dy,r,s,t))
#solution : (1,0.5)

r=1.0;s=1.0;t=1.0
x=1.0;y=2.0;dx=1.0;dy=-1.0;
println(Thetamod.AlphaThetaMax(x,dx,y,dy,r,s,t))
#doit rendre [0.381966,0.618034]
x=0.0;y=2.0;dx=1.0;dy=-2.0;
println(Thetamod.AlphaThetaMax(x,dx,y,dy,r,s,t))
#solution : [0.640388,0.697224]

#Bilan : donner une sortie bilan
if Thetamod_success==true
 println("Thetamod.jl passes the test !")
else
 println("Thetamod.jl contains some error")
end
