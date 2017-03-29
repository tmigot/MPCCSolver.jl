"""
solve(mod::MPCCmod.MPCC,r0::Float64=0.1,sigma_r::Float64=0.01,s0::Float64=0.1,sigma_s::Float64=0.01,t0::Float64=0.1,sigma_t::Float64=0.01;name_relax::AbstractString="ALAS")
solve(mod::MPCCmod.MPCC)

solve_subproblem(mod::MPCCmod.MPCC, func::Function,r::Float64,s::Float64,t::Float64,rho::Vector,name_relax::AbstractString)
solve_subproblem_ipopt(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64,rho::Vector,name_relax::AbstractString)
solve_subproblem_alas(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64,rho::Vector,name_relax::AbstractString)
"""
module MPCCsolve

using MPCCmod
using ALASMPCCmod

using Ipopt
using NLPModels
using MathProgBase
using PyPlot

"""
Methode de relaxation pour resoudre :

note : faire autrement l'initialisation de rho. (rho_init définit dans le MPCCmod)
"""
function solve(mod::MPCCmod.MPCC,r0::Float64=0.1,sigma_r::Float64=0.01,s0::Float64=0.1,sigma_s::Float64=0.01,t0::Float64=0.1,sigma_t::Float64=0.01;name_relax::AbstractString="ALAS")
#initialization
 t=t0;r=r0;s=s0;

 nb_contraintes=length(mod.mp.meta.lvar)+length(mod.mp.meta.uvar)+length(mod.mp.meta.lcon)+length(mod.mp.meta.ucon)+2*mod.nb_comp
 rho=ones(nb_contraintes) #à garder ici ?

 n=length(mod.mp.meta.x0)
 xk=mod.mp.meta.x0
 pmin=mod.prec
 realisable=MPCCmod.viol_comp(mod,xk)<=mod.prec && MPCCmod.viol_cons(mod,xk)<=mod.prec
 solved=true
 param=true

srelax_xtab=collect(xk[1:n])

#Major Loop
j=0
println("j :",j," xk :",xk[1:n]," f(xk) :",mod.mp.f(xk))
 while param && !(realisable) && solved

 # resolution du sous-problème
 if name_relax=="ALAS"
  xk,solved,s_xtab,rho = solve_subproblem(mod,solve_subproblem_alas,r,s,t,rho,name_relax)
 else
  xk,solved = solve_subproblem(mod,solve_subproblem_ipopt,r,s,t,rho,name_relax)
 end

 mod=MPCCmod.addInitialPoint(mod,xk[1:n]) #met à jour le MPCC avec le nouveau point

 r=r*sigma_r
 s=s*sigma_s
 t=t*sigma_t
 append!(srelax_xtab,collect(xk[1:n]))
 realisable=MPCCmod.viol_comp(mod,xk)<=mod.prec && MPCCmod.viol_cons(mod,xk)<=mod.prec
 param=(t+r+s)>pmin
 j+=1
println("j :",j," xk :",xk[1:n]," f(xk) :",mod.mp.f(xk))
 end

#Traitement final :
realisable || println("Infeasible solution: (comp,cons)=(",MPCCmod.viol_comp(mod,xk),",",MPCCmod.viol_cons(mod,xk),")" )
solved || println("Subproblem failure")
param || realisable || println("Parameters too small")
solved && !param && realisable && println("Success")

 # output
 return xk, mod.mp.f(xk), stat, srelax_xtab
end

"""
Méthode de relaxation avec penalisation des paramètres
"""
function solve(mod::MPCCmod.MPCC)
 solve(mod,0.1,0.01,0.1,0.01,0.1,0.01,"ALAS")
end

"""
Methode pour résoudre le sous-problème relaxé :
"""
function solve_subproblem(mod::MPCCmod.MPCC, func::Function,r::Float64,s::Float64,t::Float64,rho::Vector,name_relax::AbstractString)
 return func(mod,r,s,t,rho,name_relax) #renvoie xk,stat
end

"""
Methode pour résoudre le sous-problème relaxé :
"""
function solve_subproblem_ipopt(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64,rho::Vector,name_relax::AbstractString)
 solved=true
 nlp_relax = MPCCmod.MPCCtoRelaxNLP(mod,r,s,t,name_relax)
 output=[]

 # resolution du sous-problème avec IpOpt
 model_relax = NLPModels.NLPtoMPB(nlp_relax, IpoptSolver(print_level=0,tol=mod.prec))
 MathProgBase.optimize!(model_relax)

 if MathProgBase.status(model_relax) == :Optimal
  xk = MathProgBase.getsolution(model_relax)
  solved=false
 else
 end

 return xk,solved,output
end

"""
Methode pour résoudre le sous-problème relaxé :

note : le choix de la stratégie sur rho devrait être décidé dans mod
"""
function solve_subproblem_alas(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64,rho::Vector,name_relax::AbstractString)
 solved=true

 prec=max(s,t,r) #Il faut réflechir un peu plus sur des alternatives

 if mod.paramset.rho_restart
  alas = ALASMPCCmod.ALASMPCC(mod,r,s,t,prec) #recommence rho à chaque itération
 else
  alas = ALASMPCCmod.ALASMPCC(mod,r,s,t,prec,rho)
 end

 xk,stat,s_xtab,rho = ALASMPCCmod.solvePAS(alas) #lg,lh,lphi,s_xtab dans le output

 if stat != 0
  println("Error solve_subproblem_alas : ",stat)
  solved=false
 else
 end

 return xk,solved,s_xtab,rho
end

#end of module
end
