#Tests des packages
include("MPCCProblems.jl")

#Est-ce que l'on veut les figures :
plot=false

r0=0.1
s0=0.1
#t0=sqrt(0.1) #t=o(r) normalement
t0=0.01
sig_r=0.1
sig_s=0.1
sig_t=0.01

mpcc=ex3 #ex1, ex2, ex3

@time xb,fb,orb,nb_eval = MPCCsolve.solve(mpcc,r0,sig_r,s0,sig_s,t0,sig_t)

try
 oa=orb.inner_output_alas[1]
 @printf("%s	",orb.solve_message)
end
@show nb_eval

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
# PyPlot.title("Ex2 : min 0.5*((x1-1)^2+(x2-1)^2) s.t. 0<=x_1 _|_ x_2 >=0")
# PyPlot.title("Ex3 : min x1+x2-x3 s.t. 0<=x1 _|_ x2 >=0, x3<=4x1, x3<=4x2")
end
