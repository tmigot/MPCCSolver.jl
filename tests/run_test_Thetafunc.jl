using Thetafunc

println("On teste le package Theta")

Thetafunc_success=true
r=0.1

#theta(x::Vector,r::Float64)
#Les fonctions thetas doivent etre nulle en 0, <=1 pour x>=0 et <0 pour x<0
if Thetafunc.theta(0.0,r)!=0 && Thetafunc.theta(1.0+1e6,r)>=1 && Thetafunc(-1e-6,r)>=0
 println("Error in the definition of theta")
 plot(x->Thetafunc.theta(x,r),-0.1,2.0)
 #draw(SVG("output.svg",6inch,3inch),plot(x->Thetamod.theta(x,r),-0.1,2.0))
 Thetafunc_success=false
end

#dtheta(x::Vector,r::Float64)
#la dérivée des fonctions theta doit être croissante
if Thetafunc.theta(1.0+1e6,r)>=0 && (Thetafunc.theta(1.0,r)-Thetafunc.theta(2.0,r)>0)
 println("Error in the definition of dtheta")
 plot(x->Thetafunc.dtheta(x,r),-0.1,2.0)
 Thetafunc_success=false
end

#invtheta(x::Vector,r::Float64)
#Tester aussi les cas interdits ici pas défini pour x>1 !!!
if norm(Thetafunc.invtheta(Thetafunc.theta(1e-6,r),r)-1e-6)>precision || norm(Thetafunc.invtheta(Thetafunc.theta(-1e-6,r),r)+1e-6)>precision
 println("Error in the definition of invtheta")
 println(norm(Thetafunc.invtheta(Thetafunc.theta(1e-6,r),r)-1e-6))
 println(norm(Thetafunc.invtheta(Thetafunc.theta(-1e-6,r),r)+1e-6))
 plot(x->Thetafunc.invtheta(Thetafunc.theta(x,r),r),-0.1,1.0)
 Thetafunc_success=false
end

#AlphaThetaMax(x::Float64,dx::Float64,y::Float64,dy::Float64,r::Float64,s::Float64,t::Float64)
#AlphaThetaMax : résoud l'équation en alpha où psi(x,r,s,t)=s+t*theta(x,r)
r=1.0;s=1.0;t=0.0 #résoud l'éq.

x=1.0;y=2.0;dx=1.0;dy=-1.0;
#doit rendre (0,1)
if !isequal(Thetafunc.AlphaThetaMax(x,dx,y,dy,r,s,t),[0.0,1.0])
 println(Thetafunc.AlphaThetaMax(x,dx,y,dy,r,s,t))
 Thetafunc_success=false
end

x=0.0;y=2.0;dx=1.0;dy=-2.0;
#solution : (1,0.5)
if !isequal(Thetafunc.AlphaThetaMax(x,dx,y,dy,r,s,t),[1.0,0.5])
 println(Thetafunc.AlphaThetaMax(x,dx,y,dy,r,s,t))
 Thetafunc_success=false
end

r=1.0;s=1.0;t=1.0
#on tape la contrainte :
x=1.0;y=2.0;dx=-1.0;dy=0.0;
if !(minimum(Thetafunc.AlphaThetaMax(x,dx,y,dy,r,s,t))>=0.0) || !(maximum(Thetafunc.AlphaThetaMax(x,dx,y,dy,r,s,t))<Inf)
 println(Thetafunc.AlphaThetaMax(x,dx,y,dy,r,s,t))
 Thetafunc_success=false
end
x=0.0;y=2.0;dx=1.0;dy=-2.0;
if !(minimum(Thetafunc.AlphaThetaMax(x,dx,y,dy,r,s,t))>=0.0) || !(maximum(Thetafunc.AlphaThetaMax(x,dx,y,dy,r,s,t))<Inf)
 println(Thetafunc.AlphaThetaMax(x,dx,y,dy,r,s,t))
 Thetafunc_success=false
end

#on est sous (s,s) et donc on ne tape pas la contrainte virtuelle
x=0.0;y=0.0;dx=0.0;dy=-2.0;
if !isequal(maximum(Thetafunc.AlphaThetaMax(x,dx,y,dy,r,s,t)),Inf)
 println(Thetafunc.AlphaThetaMax(x,dx,y,dy,r,s,t))
 Thetafunc_success=false
end

#Bilan : donner une sortie bilan
if Thetafunc_success==true
 println("Theta.jl passes the test !")
else
 println("Theta.jl contains some errors")
end
