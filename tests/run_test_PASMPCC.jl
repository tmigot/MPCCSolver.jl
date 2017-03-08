#Tests des packages
include("../ALAS/MPCCmod.jl")
include("../ALAS/PASMPCC.jl")
include("../MPCC2DPlot.jl")

using PASMPCC
using MPCCmod
using Relaxationmod
using MPCC2DPlot

r=1.0;s=1.0;t=1.0;
PASMPCCmod_success=true

nlp=ADNLPModel(f,[1.0;-1.0], lvar=-Inf*ones(2), uvar=Inf*ones(2), c=c,y0=ones(1),lcon=lcon,ucon=Inf*ones(1))
mpcc = MPCCmod.MPCC(nlp,G,H,1)
alas = PASMPCC.ALASMPCC(mpcc,r,s,t)

#SlackComplementarityProjection(x::Vector,nb_comp::Int64,r::Float64,s::Float64,t::Float64)
#y1=PASMPCC.SlackComplementarityProjection([1.0;-1.0;-2.0;0.5],1,r,s,t)
#y2=PASMPCC.SlackComplementarityProjection([1.0;-1.0;1.0;-3.0],1,r,s,t)
#y3=PASMPCC.SlackComplementarityProjection([1.0;-1.0;2.0;3.0],1,r,s,t)
#if !(y1[3]>=-r && y2[4]>=-r && y3[4]==Relaxationmod.psi(y3[3],r,s,t))
# println(y1,y2)
#
# PASMPCCmod_success=false
#end

r=1.0;s=1.0;t=0.0;
nlp=ADNLPModel(f,[2.0;-1.0], lvar=-Inf*ones(2), uvar=Inf*ones(2), c=c,y0=ones(1),lcon=lcon,ucon=Inf*ones(1))
mpcc = MPCCmod.MPCC(nlp,G,H,1)
alas = PASMPCC.ALASMPCC(mpcc,r,s,t)

println("Exemple 1")
#output=PASMPCC.solvePAS(mpcc,r,s,t)
output=PASMPCC.solvePAS(alas)
println("solution :", output[1])
println("multiplicateurs :", output[2]," ",output[3]," ",output[4])
println("nb itérations :", length(output[5])/4)
nb_ite=Int(length(output[5])/4)
x1=zeros(nb_ite);x2=zeros(nb_ite)
for i=1:nb_ite
 x1[i]=output[5][4*(i-1)+1] ; x2[i]=output[5][4*(i-1)+2]
end
 PyPlot.plot(x1, x2, color="red", linewidth=2.0, linestyle="-")
 PyPlot.xlabel("x1")
 PyPlot.ylabel("x2")
 PyPlot.title("min x1-x2 s.t. x2<=1,x1>=-1,x2>=-1,(x1-1)=0 ou (x2-1)=0")

#println("Exemple 2:")
r=1.0;s=1.0;t=0.0;
nlp=ADNLPModel(f,[1.0;1.5], lvar=-Inf*ones(2), uvar=Inf*ones(2), c=c,y0=ones(1),lcon=lcon,ucon=Inf*ones(1))
mpcc = MPCCmod.MPCC(nlp,G,H,1)
alas = PASMPCC.ALASMPCC(mpcc,r,s,t)
#output=PASMPCC.solvePAS(mpcc,r,s,t)

# Un exemple simple pour tester :
#println("Exemple 3:")
# f2(x)=x[1]^2-x[1]*x[2]+1.0/3.0*x[2]^2-2*x[1]
# G2(x)=x[1]
# H2(x)=x[2]
# cc(x)=[x[1]+10]
# lcon2=zeros(1)
r=1.0;s=1.0;t=0.0;
#nlp2=ADNLPModel(f2,[-1.0;1.0], lvar=-Inf*ones(2), uvar=Inf*ones(2), c=cc,y0=ones(1),lcon=lcon2,ucon=Inf*ones(1))
#mpcc2 = MPCCmod.MPCC(nlp2,G2,H2,1)
#alas = PASMPCC.ALASMPCC(mpcc,r,s,t)
#output=PASMPCC.solvePAS(mpcc2,r,s,t)

#MPCC2DPlot.RelaxationPlot([1.0 -1.0;1.0 1.5],r,s,t,-2.0,2.0)

#Bilan : donner une sortie bilan
if PASMPCCmod_success==true
 println("PASMPCC.jl passes the test !")
else
 println("PASMPCC.jl contains some error")
end
