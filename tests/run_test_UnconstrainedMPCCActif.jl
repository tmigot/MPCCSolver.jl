#Tests des packages
include("../ALAS/UnconstrainedMPCCActif.jl")

using UnconstrainedMPCCActif
using ActifMPCCmod

UnconstrainedMPCCActif_success=true

r=1.0;s=1.0;t=0.0; #relaxation KDB

nlp=ADNLPModel(f,ones(2), lvar=-Inf*ones(2), uvar=Inf*ones(2), c=c,y0=ones(1),lcon=lcon,ucon=Inf*ones(1))
mpcc = MPCCmod.MPCC(nlp,G,H,1)
nlp_ma=ADNLPModel(x->mpcc.mp.f(x)+norm(G(x)-x[3])^2+norm(H(x)-x[4])^2, [-1.0;1.0;-r;1.0],lvar=[mpcc.mp.meta.lvar;-r*ones(2*mpcc.nb_comp)], uvar=[mpcc.mp.meta.uvar;Inf*ones(2*mpcc.nb_comp)])
ma=ActifMPCCmod.MPCC_actif(nlp_ma,r,s,t,1)

x0=[-1.0;1.0]
#grad : [1+2(x1-x3);-1+2(x2-x4);-2(x1-x3);-2(x2-x4)]=[1;-1;0;0]
if !(UnconstrainedMPCCActif.SteepestDescent(ma,x0)==[-1;1])
 println("SteepestDescent")
 println(UnconstrainedMPCCActif.SteepestDescent(ma,x0))

 UnconstrainedMPCCActif_success=false
end

output=UnconstrainedMPCCActif.LineSearchSolve(ma,x0,0.0,zeros(x0))
if !(output[1]==[-1.5;1.5;-1.0;1.0])
 println("test 1 : [xjp,w,dj]")
 println(output[1])
end

#Bilan : donner une sortie bilan
if UnconstrainedMPCCActif_success==true
 println("UnconstrainedMPCCActif.jl passes the test !")
else
 println("UnconstrainedMPCCActif.jl contains some error")
end
