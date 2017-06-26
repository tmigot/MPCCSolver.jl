gc()

#Tests des packages
include("../ALAS/include.jl")

using MPCCmod
using OutputRelaxationmod
#package pour plot
using PyPlot

using NLPModels

#Est-ce que l'on veut les figures :
plot=false

# Exemple 1 :
 nb_comp=1
 f(x)=x[1]-x[2]
 Gfunc(x)=x[1]
 Hfunc(x)=x[2]
 c(x)=[1-x[2]]
 lcon=[0.0]
 ucon=[Inf]
 G=SimpleNLPModel(f, collect([1.0,1.0]), lvar=-Inf*ones(2), uvar=Inf*ones(2), c=Gfunc,  lcon=zeros(nb_comp), ucon=Inf*ones(nb_comp), J=x->[1.0 0.0], Jtp=(x,v)->[v[1], 0.0])
 H=SimpleNLPModel(f, collect([1.0,1.0]), lvar=-Inf*ones(2), uvar=Inf*ones(2), c=Hfunc,  lcon=zeros(nb_comp), ucon=Inf*ones(nb_comp), J=x->[0.0 1.0], Jtp=(x,v)->[0.0, v[1]])

r0=0.1
s0=0.1
#t0=sqrt(0.1) #t=o(r) normalement
t0=0.0
sig_r=0.1
sig_s=0.1
sig_t=0.01

#déclare le mpcc :
#ex1= MPCCmod.MPCC(f,ones(2),Gfunc,Hfunc,1,-Inf*ones(2),Inf*ones(2),c,lcon,ucon)
ex1= MPCCmod.MPCC(f,ones(2),G,H,1,-Inf*ones(2),Inf*ones(2),c,lcon,ucon)

println("\n \n ALAS Exemple 1 :")
#println("KDB method:")
#résolution avec ALAS KDB
#xkdb,fkdb,statkdb = MPCCsolve.solve(ex1,r0,sig_r,s0,sig_s,0.0,sig_t)
println("Butterfly method:")
#résolution avec ALAS Butterfly
xb,fb,orb = MPCCsolve.solve(ex1,r0,sig_r,s0,sig_s,t0,sig_t)

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

break

# Exemple 2 :
 f(x)=0.5*((x[1]-1)^2+(x[2]-1)^2)
 Gfunc(x)=x[1]
 nb_comp=1
 Hfunc(x)=x[2]
 G=SimpleNLPModel(f, collect([1.0,1.0]), lvar=-Inf*ones(2), uvar=Inf*ones(2), c=Gfunc,  lcon=zeros(nb_comp), ucon=Inf*ones(nb_comp), J=x->[1.0 0.0])
 H=SimpleNLPModel(f, collect([1.0,1.0]), lvar=-Inf*ones(2), uvar=Inf*ones(2), c=Hfunc,  lcon=zeros(nb_comp), ucon=Inf*ones(nb_comp), J=x->[0.0 1.0])
 c(x)=zeros(1)
 lcon=zeros(1)
 ucon=ones(1)

r0=0.1
s0=0.1
t0=sqrt(r0)
sig_r=0.1
sig_s=0.1
sig_t=0.01
println("\n \n ALAS Exemple 2 :")

#déclare le mpcc :
ex2= MPCCmod.MPCC(f,collect([1.0,1.0]),G,H,1,-Inf*ones(2),Inf*ones(2),c,lcon,ucon)
#println("KDB method:")
#résolution avec ALAS KDB
#xkdb,fkdb,statkdb = MPCCsolve.solve(ex2,r0,sig_r,s0,sig_s,0.0,sig_t)
println("Butterfly method:")
#résolution avec ALAS Butterfly
xb,fb,orb = MPCCsolve.solve(ex2,r0,sig_r,s0,sig_s,t0,sig_t)

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

# Exemple 3 :
# minimize x1+x2-x3
# s.t.     0<=x1 _|_ x2>=0
#          -4x1+x3<=0
#          -4x2+x3<=0
 f(x)=x[1]+x[2]-x[3]
 Gfunc(x)=[x[1]+0*x[2]+0*x[3]]
 Hfunc(x)=[x[2]+0*x[1]+0*x[3]]
 c(x)=[-4*x[1]+x[3];-4*x[2]+x[3]]
 lcon=-Inf*ones(2)
 ucon=zeros(2)
 G=SimpleNLPModel(f, collect([1.0,1.0]), lvar=-Inf*ones(2), uvar=Inf*ones(2), c=Gfunc,  lcon=zeros(nb_comp), ucon=Inf*ones(nb_comp), J=x->[1.0 0.0])
 H=SimpleNLPModel(f, collect([1.0,1.0]), lvar=-Inf*ones(2), uvar=Inf*ones(2), c=Hfunc,  lcon=zeros(nb_comp), ucon=Inf*ones(nb_comp), J=x->[0.0 1.0])

r0=0.1
s0=0.1
t0=0.01
sig_r=0.1
sig_s=0.1
sig_t=0.01

println("\n \n ALAS Exemple 3 :")

#déclare le mpcc :
#ex3= MPCCmod.MPCC(f,[0.5;1.0;0.0],G,H,1,-Inf*ones(3),Inf*ones(3),c,ones(2),lcon,ucon)
#println("KDB method:")
#résolution avec ALAS KDB
#xkdb,fkdb,orkdb = MPCCsolve.solve(ex3,r0,sig_r,s0,sig_s,0.0,sig_t)
ex3= MPCCmod.MPCC(f,[0.5;1.0;0.0],G,H,1,-Inf*ones(3),Inf*ones(3),c,lcon,ucon)
println("Butterfly method:")
#résolution avec ALAS Butterfly
xb,fb,orb = MPCCsolve.solve(ex3,r0,sig_r,s0,sig_s,t0,sig_t)

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
