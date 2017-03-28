using UnconstrainedMPCCActif
using ActifMPCCmod

UnconstrainedMPCCActif_success=true
println("On teste le module UnconstrainedMPCCActif")

r=1.0;s=1.0;t=0.0; #relaxation KDB

nlp=ADNLPModel(f,ones(2), lvar=-Inf*ones(2), uvar=Inf*ones(2), c=c,y0=ones(1),lcon=lcon,ucon=Inf*ones(1))
mpcc = MPCCmod.MPCC(nlp,G,H,1)
nlp_ma=ADNLPModel(x->mpcc.mp.f(x)+norm(G(x)-x[3])^2+norm(H(x)-x[4])^2, [-1.0;1.0;-r;1.0],lvar=[mpcc.mp.meta.lvar;-r*ones(2*mpcc.nb_comp)], uvar=[mpcc.mp.meta.uvar;Inf*ones(2*mpcc.nb_comp)])
ma=ActifMPCCmod.MPCC_actif(nlp_ma,r,s,t,1)



#Bilan : donner une sortie bilan
if UnconstrainedMPCCActif_success==true
 println("UnconstrainedMPCCActif.jl passes the test !")
else
 println("UnconstrainedMPCCActif.jl contains some error")
end
