#Tests des packages
include("../ALAS/include.jl")
include("../MPCCProblems/MPCCProblems.jl")
include("../ALAS/MPCCPlot2D.jl")

using MPCCProblems
using OutputRelaxationmod
#package pour plot
using PyPlot
#Est-ce que l'on veut les figures :
plot=false

#Est-ce que l'on veut les figures :
plot=false

r0=1.0
s0=0.5
#t0=sqrt(0.1) #t=o(r) normalement
t0=0.5
sig_r=0.1
sig_s=0.1
sig_t=0.05

mpcc=MPCCProblems.ex1bd() #ex1, ex2, ex3

@time xb,rmpcc,orb = MPCCSolvemod.solve(mpcc)

try
 oa=orb.inner_output_alas[1]
 @printf("%s	\n",orb.solve_message)
end
@show orb.nb_eval
"Hiha !"
#IterationsPlot2D(1,2,n+1,n+2,orb)
