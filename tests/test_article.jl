gc()

#Tests des packages
include("../ALAS/include.jl")

using MPCCmod
using OutputRelaxationmod
#package pour plot
using PyPlot

using NLPModels
using JuMP

#Est-ce que l'on veut les figures :
plot=false

println("\n \n ALAS Exemple 1 :")

r0=0.1
s0=0.1
#t0=sqrt(0.1) #t=o(r) normalement
t0=0.01
sig_r=0.1
sig_s=0.1
sig_t=0.01

ex1=JuMP.Model()
JuMP.@variable(ex1,x[1:2],start=1.0)
JuMP.@NLobjective(ex1,Min,x[1]-x[2])
JuMP.@constraint(ex1,1-x[2]>=0)
ex1=MathProgNLPModel(ex1)
G=JuMP.Model()
JuMP.@variable(G,x[1:2],start=1.0)
JuMP.@constraint(G,x[1]>=0)
JuMP.@NLobjective(G,Min,0.0)
G=MathProgNLPModel(G)
H=JuMP.Model()
JuMP.@variable(H,x[1:2],start=1.0)
JuMP.@constraint(H,x[2]>=0)
JuMP.@NLobjective(H,Min,0.0)
H=MathProgNLPModel(H)

@time ex1= MPCCmod.MPCC(ex1,G,H)


println("Butterfly method:")
#résolution avec ALAS Butterfly
@time xb,fb,orb = MPCCsolve.solve(ex1,r0,sig_r,s0,sig_s,t0,sig_t)


 
if plot
 subplot(311)
 suptitle("Quadratic penalty and Newton spectral")
 nb_ite=length(orb.rtab)
 x1=zeros(nb_ite);x2=zeros(nb_ite)
 x1=collect(orb.xtab[1,1:nb_ite])
 x2=collect(orb.xtab[2,1:nb_ite])
 PyPlot.plot(x1, x2, color="red",marker="*")
 PyPlot.xlabel("x1")
 PyPlot.ylabel("x2")
 PyPlot.title("Ex1 : min x1-x2 s.t. 0<=x_1 _|_ x_2 >=0")
end

# Exemple 2 :
println("\n \n ALAS Exemple 2 :")
ex2=JuMP.Model()
JuMP.@variable(ex2,x[1:2],start=1.0)
JuMP.@NLobjective(ex2,Min,0.5*((x[1]-1)^2+(x[2]-1)^2))
ex2=MathProgNLPModel(ex2)
G=JuMP.Model()
JuMP.@variable(G,x[1:2],start=1.0)
JuMP.@constraint(G,x[1]>=0)
JuMP.@NLobjective(G,Min,0.0)
G=MathProgNLPModel(G)
H=JuMP.Model()
JuMP.@variable(H,x[1:2],start=1.0)
JuMP.@constraint(H,x[2]>=0)
JuMP.@NLobjective(H,Min,0.0)
H=MathProgNLPModel(H)

@time ex2= MPCCmod.MPCC(ex2,G,H)

r0=0.1
s0=0.1
t0=sqrt(r0)
sig_r=0.1
sig_s=0.1
sig_t=0.01


#déclare le mpcc :
#ex2= MPCCmod.MPCC(f,collect([1.0,1.0]),G,H,1,-Inf*ones(2),Inf*ones(2),c,lcon,ucon)
#println("KDB method:")
#résolution avec ALAS KDB
#xkdb,fkdb,statkdb = MPCCsolve.solve(ex2,r0,sig_r,s0,sig_s,0.0,sig_t)
println("Butterfly method:")
#résolution avec ALAS Butterfly
@time xb,fb,orb = MPCCsolve.solve(ex2,r0,sig_r,s0,sig_s,t0,sig_t)

if plot
 subplot(312)
 #nb_ite=Int(length(s_xtab)/2)
 #x1=zeros(nb_ite);x2=zeros(nb_ite)
 #for i=1:nb_ite
 # x1[i]=s_xtab[2*(i-1)+1] ; x2[i]=s_xtab[2*(i-1)+2]
 #end
 x1=collect(orb.xtab[1,1:length(orb.rtab)])
 x2=collect(orb.xtab[2,1:length(orb.rtab)])
 #PyPlot.plot(x1, x2, color="red", linewidth=2.0, linestyle="-")
 PyPlot.plot(x1, x2, color="red",marker="*")
 PyPlot.xlabel("x1")
 PyPlot.ylabel("x2")
 PyPlot.title("Ex2 : min 0.5*((x1-1)^2+(x2-1)^2) s.t. 0<=x_1 _|_ x_2 >=0")
end

println("\n \n ALAS Exemple 3 :")
# Exemple 3 :
# minimize x1+x2-x3
# s.t.     0<=x1 _|_ x2>=0
#          -4x1+x3<=0
#          -4x2+x3<=0

ex3=JuMP.Model()
JuMP.@variable(ex3,x[1:3],start=1.0)
JuMP.@NLobjective(ex3,Min,x[1]+x[2]-x[3])
JuMP.@NLconstraint(ex3,c1,-4*x[1]+x[3]<=0)
JuMP.@NLconstraint(ex3,c2,-4*x[2]+x[3]<=0)
ex3=MathProgNLPModel(ex3)
G=JuMP.Model()
JuMP.@variable(G,x[1:3],start=1.0)
JuMP.@constraint(G,x[1]>=0)
JuMP.@NLobjective(G,Min,0.0)
G=MathProgNLPModel(G)
H=JuMP.Model()
JuMP.@variable(H,x[1:3],start=1.0)
JuMP.@constraint(H,x[2]>=0)
JuMP.@NLobjective(H,Min,0.0)
H=MathProgNLPModel(H)

@time ex3= MPCCmod.MPCC(ex3,G,H)

r0=0.1
s0=0.1
t0=0.01
sig_r=0.1
sig_s=0.1
sig_t=0.01

#déclare le mpcc :
#ex3= MPCCmod.MPCC(f,[0.5;1.0;0.0],G,H,1,-Inf*ones(3),Inf*ones(3),c,ones(2),lcon,ucon)
#println("KDB method:")
#résolution avec ALAS KDB
#xkdb,fkdb,orkdb = MPCCsolve.solve(ex3,r0,sig_r,s0,sig_s,0.0,sig_t)
#ex3= MPCCmod.MPCC(f,[0.5;1.0;0.0],G,H,1,-Inf*ones(3),Inf*ones(3),c,lcon,ucon)
println("Butterfly method:")
#résolution avec ALAS Butterfly
@time xb,fb,orb = MPCCsolve.solve(ex3,r0,sig_r,s0,sig_s,t0,sig_t)

if plot
 subplot(313)
 #nb_ite=Int(length(s_xtab)/3)
 #x1=zeros(nb_ite);x2=zeros(nb_ite);x3=zeros(nb_ite)
 #for i=1:nb_ite
 # x1[i]=s_xtab[3*(i-1)+1] ; x2[i]=s_xtab[3*(i-1)+2] ; x3[i]=s_xtab[3*(i-1)+3]
 #end
 x1=collect(orb.xtab[1,1:length(orb.rtab)])
 x2=collect(orb.xtab[2,1:length(orb.rtab)])
 #PyPlot.plot(x1, x3, color="red", linewidth=2.0, linestyle="-")
 PyPlot.plot(x1, x2, color="red",marker="*")
 PyPlot.xlabel("x1")
 PyPlot.ylabel("x3")
 PyPlot.title("Ex3 : min x1+x2-x3 s.t. 0<=x1 _|_ x2 >=0, x3<=4x1, x3<=4x2")
end
