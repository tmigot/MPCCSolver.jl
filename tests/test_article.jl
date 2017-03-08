gc()
#Tests des packages
include("../ALAS/Thetamod.jl")
include("../ALAS/Relaxationmod.jl")
include("../ALAS/ActifMPCCmod.jl")
include("../ALAS/UnconstrainedMPCCActif.jl")
include("../ALAS/MPCCmod.jl")
include("../ALAS/PASMPCC.jl")
include("../ALAS/MPCCsolve.jl")
include("../MPCC2DPlot.jl")

using MPCCmod
#package pour plot
using PyPlot

prec=1e-4

# Exemple 1 :
 f(x)=x[1]-x[2]
 G(x)=x[1]
 H(x)=x[2]
 c(x)=[1-x[2]]
 lcon=zeros(1)

r0=0.1
s0=0.1
#t0=sqrt(0.1) #t=o(r) normalement
t0=0.01
sig_r=0.1
sig_s=0.1
sig_t=0.01

#déclare le mpcc :
ex1= MPCCmod.MPCC(f,ones(2),G,H,1,-Inf*ones(2),Inf*ones(2),c,ones(1),lcon,Inf*ones(1),prec)

println("ALAS Exemple 1 :")
println("KDB method:")
#résolution avec ALAS KDB
xkdb,fkdb,statkdb = MPCCsolve.solve(ex1,r0,sig_r,s0,sig_s,0.0,sig_t)
println("Butterfly method:")
#résolution avec ALAS Butterfly
xb,fb,statb, s_xtab = MPCCsolve.solve(ex1,r0,sig_r,s0,sig_s,t0,sig_t)

if false
 nb_ite=Int(length(s_xtab)/2)
 x1=zeros(nb_ite);x2=zeros(nb_ite)
 for i=1:nb_ite
  x1[i]=s_xtab[2*(i-1)+1] ; x2[i]=s_xtab[2*(i-1)+2]
 end
 PyPlot.plot(x1, x2, color="red", linewidth=2.0, linestyle="-")
 PyPlot.xlabel("x1")
 PyPlot.ylabel("x2")
 PyPlot.title("min x1-x2 s.t. x2<=1,x1>=-1,x2>=-1,(x1-1)=0 ou (x2-1)=0")
end

# Exemple 2 :
 f(x)=0.5*((x[1]-1)^2+(x[2]-1)^2)
 G(x)=x[1]
 H(x)=x[2]
 c(x)=zeros(1)
 lcon=zeros(1)
 ucon=ones(1)

r0=0.1
s0=0.1
t0=sqrt(r0)
sig_r=0.1
sig_s=0.1
sig_t=0.01

println("ALAS Exemple 2 :")

#déclare le mpcc :
ex2= MPCCmod.MPCC(f,ones(2),G,H,1,-Inf*ones(2),Inf*ones(2),c,ones(1),lcon,ucon,prec)
println("KDB method:")
#résolution avec ALAS KDB
xkdb,fkdb,statkdb = MPCCsolve.solve(ex2,r0,sig_r,s0,sig_s,0.0,sig_t)
println("Butterfly method:")
#résolution avec ALAS Butterfly
xb,fb,statb, s_xtab = MPCCsolve.solve(ex2,r0,sig_r,s0,sig_s,t0,sig_t)

if false
 nb_ite=Int(length(s_xtab)/2)
 x1=zeros(nb_ite);x2=zeros(nb_ite)
 for i=1:nb_ite
  x1[i]=s_xtab[2*(i-1)+1] ; x2[i]=s_xtab[2*(i-1)+2]
 end
 PyPlot.plot(x1, x2, color="red", linewidth=2.0, linestyle="-")
 PyPlot.xlabel("x1")
 PyPlot.ylabel("x2")
 PyPlot.title("min 0.5*((x1-1)^2+(x2-1)^2) s.t. 0<=x_1 _|_ x_2 >=0")
end

# Exemple 3 :
# minimize x1+x2-x3
# s.t.     0<=x1 _|_ x2>=0
#          -4x1+x3<=0
#          -4x2+x3<=0
 f(x)=x[1]+x[2]-x[3]
 G(x)=x[1]
 H(x)=x[2]
 c(x)=[-4*x[1]+x[3];-4*x[2]+x[3]]
 lcon=-Inf*ones(2)
 ucon=zeros(2)

r0=0.1
s0=0.1
t0=0.01
sig_r=0.1
sig_s=0.1
sig_t=0.01

println("ALAS Exemple 3 :")

#déclare le mpcc :
ex3= MPCCmod.MPCC(f,ones(3),G,H,1,-Inf*ones(3),Inf*ones(3),c,ones(2),lcon,ucon,prec)
println("KDB method:")
#résolution avec ALAS KDB
xkdb,fkdb,statkdb = MPCCsolve.solve(ex3,r0,sig_r,s0,sig_s,0.0,sig_t)
println("Butterfly method:")
#résolution avec ALAS Butterfly
#xb,fb,statb, s_xtab = MPCCsolve.solve(ex3,r0,sig_r,s0,sig_s,t0,sig_t)

if false
 nb_ite=Int(length(s_xtab)/2)
 x1=zeros(nb_ite);x2=zeros(nb_ite)
 for i=1:nb_ite
  x1[i]=s_xtab[2*(i-1)+1] ; x2[i]=s_xtab[2*(i-1)+2]
 end
 PyPlot.plot(x1, x2, color="red", linewidth=2.0, linestyle="-")
 PyPlot.xlabel("x1")
 PyPlot.ylabel("x2")
 PyPlot.title("min x_1+x_2-x_3 s.t. \n 0<=x_1 _|_ x_2 >=0")
end
