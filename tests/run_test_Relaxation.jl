print_with_color(:yellow, "Test Relaxation module:\n")

using Relaxation

r=1.0;s=1.0;t=1.0;

Relaxation_success = true

# psi(x::Any,r::Float64,s::Float64,t::Float64)
t0 = norm(Relaxation.psi(s,r,s,t)-s) <= eps(Float64)*10

if !t0

 x = linspace(-0.5,10,1000); y = Relaxation.psi(x,r,s,t)
 PyPlot.plot(x, y, color="red", linewidth=2.0, linestyle="-")
 x = linspace(-0.5,10,1000); y = Relaxation.psi(x,r,s,t)
 PyPlot.plot(y, x, color="blue", linewidth=2.0, linestyle="-")

 Relaxation_success=false
 throw("UnitaryTestFailure: Error in the definition of psi")

end

#invpsi(x::Any,r::Float64,s::Float64,t::Float64)
t1 = norm(Relaxation.invpsi(s,r,s,t)-s) <= eps(Float64)*10

if !t1

 x = linspace(-0.5,10,1000); y = Relaxation.psi(x,r,s,t)
 PyPlot.plot(x, y, color="red", linewidth=2.0, linestyle="-")
 x = linspace(-0.5,10,1000); y = Relaxation.psi(x,r,s,t)
 PyPlot.plot(y, x, color="blue", linewidth=2.0, linestyle="-")

 Relaxation_success=false
 throw("UnitaryTestFailure: Error in the definition of invpsi")

end
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
t2 = isequal(Relaxation.alpha_max(x,dx,y,dy,r,s,t),[0.0,1.0])

if !t2

 println(Relaxation.alpha_max(x,dx,y,dy,r,s,t))
 Relaxation_success = false

end

x=0.0;y=2.0;dx=1.0;dy=-2.0;
#solution : (1,0.5)
t3 = isequal(Relaxation.alpha_max(x,dx,y,dy,r,s,t),[1.0,0.5])

if !t3

 println(Relaxation.alpha_max(x,dx,y,dy,r,s,t))
 Relaxation_success = false

end

r=1.0;s=1.0;t=1.0
#on tape la contrainte :
x=1.0;y=2.0;dx=1.0;dy=0.0;

t4 = minimum(Relaxation.alpha_max(x,dx,y,dy,r,s,t))>=0.0
t5 = maximum(Relaxation.alpha_max(x,dx,y,dy,r,s,t))==Inf

if !(t4) || !(t5)

 println(Relaxation.alpha_max(x,dx,y,dy,r,s,t))
 Relaxation_success = false

end

x=2.0;y=1.0;dx=-1.0;dy=2.0;

t6 = minimum(Relaxation.alpha_max(x,dx,y,dy,r,s,t)) >= 0.0
t7 = maximum(Relaxation.alpha_max(x,dx,y,dy,r,s,t)) <  Inf

if !(t6) || !(t7)

 println(Relaxation.alpha_max(x,dx,y,dy,r,s,t))
 Relaxation_success = false

end

#on est sous (s,s) et donc on ne tape pas la contrainte virtuelle
x=0.0;y=0.0;dx=0.0;dy=-2.0;

t8 = isequal(maximum(Relaxation.alpha_max(x,dx,y,dy,r,s,t)),Inf)

if !t8

 println(Relaxation.alpha_max(x,dx,y,dy,r,s,t))
 Relaxation_success = false

end

#Conclusion:
if Relaxation_success==true
 print_with_color(:yellow, "Relaxationmod.jl passes the test !\n")
else
 print_with_color(:red, "Relaxationmod.jl contains some error\n")
end
