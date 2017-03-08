#Tests des packages
include("../ALAS/MPCCmod.jl")
include("../ALAS/PASMPCC.jl")
include("../ALAS/MPCCsolve.jl")

using MPCCmod

MPCCmod_success=true
f(x)=x[1]-x[2]

#On transforme la sortie NL en type MPCC :
#Constructeur 1:
mpcc1 = MPCCmod.MPCC(f,ones(2),G,H,1,-Inf*ones(2),Inf*ones(2),c,ones(1),lcon,Inf*ones(1))

#Constructeur 2:
nlp=ADNLPModel(f,ones(2), lvar=-Inf*ones(2), uvar=Inf*ones(2), c=c,y0=ones(1),lcon=lcon,ucon=Inf*ones(1))
mpcc = MPCCmod.MPCC(nlp,G,H,1,1e-6)

println("")
println("KDB test")
println("")
xkdb,fkdb,statkdb = MPCCsolve.solve(mpcc,0.1,0.01,0.1,0.01,0.0,0.01,"KDB")

if false
#Test le solve
xss,fss,statss = MPCCsolve.solve(mpcc)
xb,fb,statb = MPCCsolve.solve(mpcc,0.1,0.01,0.1,0.01,0.1,0.01,"Butterfly")
println("")
println("KDB test")
println("")
xkdb,fkdb,statkdb = MPCCsolve.solve(mpcc,0.1,0.01,0.1,0.01,0.0,0.01,"KDB")
xks,fks,statks = MPCCsolve.solve(mpcc,0.1,0.01,0.1,0.01,0.1,0.01,"KS")

if !(norm(-1-fss)<1e-6 || norm(-1-fb)<1e-6 || norm(-1-fkdb)<1e-6 || norm(-1-fks)<1e-6)
 println("Distance à l'objectif")
 println(norm(-1-fss)<1e-6, norm(-1-fb)<1e-6 , norm(-1-fkdb)<1e-6, norm(-1-fks)<1e-6)
 MPCCmod_success=false
end
if !(MPCCmod.viol_comp(mpcc,xss)<1e-6 || MPCCmod.viol_comp(mpcc,xb)<1e-6 || MPCCmod.viol_comp(mpcc,xkdb)<1e-6 || MPCCmod.viol_comp(mpcc,xks)<1e-6)
 println("Satisfaction de la complémentarité")
 println(MPCCmod.viol_comp(mpcc,xss)<1e-6,MPCCmod.viol_comp(mpcc,xb)<1e-6,MPCCmod.viol_comp(mpcc,xkdb)<1e-6,MPCCmod.viol_comp(mpcc,xks)<1e-6)
 MPCCmod_success=false
end
end

#Bilan : donner une sortie bilan
if MPCCmod_success==true
 println("MPCCmod.jl passes the test !")
else
 println("MPCCmod.jl contains some error")
end
