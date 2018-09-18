using ThetaFct

print_with_color(:yellow, "Test Theta module:\n")

ThetaFct_success = true

r=0.1

#theta(x::FloatOrVector,r::Float64)
#theta's vanishe in 0, are bounded by 1, increasing, concave
# positive for positive values, and negative for negative values.
t0 = ThetaFct.theta(0.0, r) .== 0.0
t1 = ThetaFct.theta(1.0+1e6, r) .<= 1
t2 = ThetaFct.theta(-1e-6, r) .< 0.0
t3 = ThetaFct.theta(1.0, r) .>= ThetaFct.theta(0.5, r)

if !(t0 && t1 && t2 && t3)

 throw("UnitaryTestFailure: Error in the definition of theta")
 #plot(x->ThetaFct.theta(x,r),-0.1,2.0)
 #draw(SVG("output.svg",6inch,3inch),plot(x->Thetamod.theta(x,r),-0.1,2.0))
 ThetaFct_success = false

end

#dtheta(x::FloatOrVector,r::Float64)
#la dérivée des fonctions theta doit être positive
t4 = ThetaFct.dtheta(1.0+1e6, r) >= 0.0
t5 = ThetaFct.dtheta(-0.5, r)    >= 0.0
if !(t4 && t5)

 throw("UnitaryTestFailure: Error in the definition of dtheta")
 #plot(x->ThetaFct.dtheta(x,r),-0.1,2.0)
 ThetaFct_success = false

end

#ddtheta(x::FloatOrVector,r::Float64)
#la dérivée seconde des fonctions theta doit être négative
t6 = ThetaFct.ddtheta(1.0+1e6, r) <= 0.0
t7 = ThetaFct.ddtheta(-0.5, r)    <= 0.0

if !(t6 && t7)

 throw("UnitaryTestFailure: Error in the definition of ddtheta")
 #plot(x->ThetaFct.dtheta(x,r),-0.1,2.0)
 ThetaFct_success = false

end

#invtheta(x::Vector,r::Float64)
t8 = abs(ThetaFct.invtheta(ThetaFct.theta(1.,r),r)-1.)      <= eps(Float64)*10.
t9 = abs(ThetaFct.invtheta(ThetaFct.theta(-1e-6,r),r)+1e-6) <= eps(Float64)*10.

if !(t8 && t9)

 throw("UnitaryTestFailure: Error in the definition of invtheta")
 #plot(x->ThetaFct.invtheta(ThetaFct.theta(x,r),r),-0.1,1.0)
 ThetaFct_success = false

end

#_poly_degree_2_positif
#Compute the smallest positive root of a second order polynomial
#or a negative/complex value if no positive real root.
#If no solution or infinite nb of solutions, return Inf.
t10 = abs(ThetaFct._poly_degree_2_positif(0.5,-1.,0.25) - 0.2928932188134524) <= eps(Float64)
t11 = abs(ThetaFct._poly_degree_2_positif(0.,1.,-0.5) - 0.5) <= eps(Float64)
t12 = !isfinite(ThetaFct._poly_degree_2_positif(0.,0.,-0.))
t13 = !isfinite(ThetaFct._poly_degree_2_positif(1.,1.,1.))

if !(t10 && t11 & t12 & t13)

 throw("UnitaryTestFailure: Error in the definition of _poly_degree_2_positif")
 ThetaFct_success = false

end

#alpha_theta_max(x::Float64,dx::Float64,y::Float64,dy::Float64,r::Float64,s::Float64,t::Float64)
#alpha_theta_max : résoud l'équation en alpha où psi(x,r,s,t)=s+t*theta(x,r)
r=1.0;s=1.0;t=0.0 #résoud l'éq.

x=1.0;y=2.0;dx=1.0;dy=-1.0;
#doit rendre (0,1)
t14 = isequal(ThetaFct.alpha_theta_max(x,dx,y,dy,r,s,t), [0.0,1.0])
if !t14

 println(ThetaFct.alpha_theta_max(x,dx,y,dy,r,s,t))
 ThetaFct_success = false

end

x=0.0;y=2.0;dx=1.0;dy=-2.0;
#solution : (1,0.5)

t15 = isequal(ThetaFct.alpha_theta_max(x,dx,y,dy,r,s,t), [1.0,0.5])
if !t15

 println(ThetaFct.alpha_theta_max(x,dx,y,dy,r,s,t))
 ThetaFct_success = false

end

r=1.0;s=1.0;t=1.0
#on tape la contrainte :
x=1.0;y=2.0;dx=1.0;dy=0.0;

t16 = minimum(ThetaFct.alpha_theta_max(x,dx,y,dy,r,s,t))>=0.0
t17 = maximum(ThetaFct.alpha_theta_max(x,dx,y,dy,r,s,t))==Inf

if !(t16) || !(t17)

 println(ThetaFct.alpha_theta_max(x,dx,y,dy,r,s,t))
 ThetaFct_success = false

end

x=2.0;y=1.0;dx=-1.0;dy=2.0;

t18 = minimum(ThetaFct.alpha_theta_max(x,dx,y,dy,r,s,t))>=0.0
t19 = maximum(ThetaFct.alpha_theta_max(x,dx,y,dy,r,s,t))<Inf

if !(t18) || !(t19)

 println(ThetaFct.alpha_theta_max(x,dx,y,dy,r,s,t))
 ThetaFct_success = false

end

#on est sous (s,s) et donc on ne tape pas la contrainte virtuelle
x=0.0;y=0.0;dx=0.0;dy=-2.0;

t20 = isequal(maximum(ThetaFct.alpha_theta_max(x,dx,y,dy,r,s,t)),Inf)

if !t20

 println(ThetaFct.alpha_theta_max(x,dx,y,dy,r,s,t))
 ThetaFct_success = false

end

#Conclusion:
if ThetaFct_success==true
 print_with_color(:yellow, "Theta.jl passes the test !\n")
end
