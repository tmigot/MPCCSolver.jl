#Tests des packages
include("MPCCProblems.jl")
include("../ALAS/MPCCPlot2D.jl")

#Est-ce que l'on veut les figures :
plot=false

r0=1.0
s0=0.5
#t0=sqrt(0.1) #t=o(r) normalement
t0=0.5
sig_r=0.1
sig_s=0.1
sig_t=0.05

mpcc=ex1 #ex1, ex2, ex3

#@time xb,fb,orb,nb_eval = MPCCsolve.solve(mpcc,r0,sig_r,s0,sig_s,t0,sig_t)
@time xb,orb,nb_eval = MPCCsolve.solve(mpcc)
#mpcc.mp.meta.x0

try
 oa=orb.inner_output_alas[1]
 @printf("%s	",orb.solve_message)
end
@show nb_eval

nbc=mpcc.nb_comp;n=length(xb)-2*nbc

#IterationsPlot2D(1,2,n+1,n+2,orb)
