"""
solve_subproblem(mod::MPCCmod.MPCC, func::Function,r::Float64,s::Float64,t::Float64,name_relax::AbstractString)
solve_subproblem_ipopt(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64,name_relax::AbstractString)
solve_subproblem_alas(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64,name_relax::AbstractString)

solve(mod::MPCCmod.MPCC,r0::Float64,sigma_r::Float64,s0::Float64,sigma_s::Float64,t0::Float64,sigma_t::Float64,name_relax::AbstractString)
solve(mod::MPCCmod.MPCC)
solve(mod::MPCCmod.MPCC,r0::Float64,sigma_r::Float64,s0::Float64,sigma_s::Float64,t0::Float64,sigma_t::Float64)
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
"""
function solve(mod::MPCCmod.MPCC,r0::Float64,sigma_r::Float64,s0::Float64,sigma_s::Float64,t0::Float64,sigma_t::Float64,name_relax::AbstractString)
#initialization
 t=t0;r=r0;s=s0;
 n=length(mod.mp.meta.x0)
 xk=mod.mp.meta.x0 #-- peut-être pas besoin de la variable xk ?
 pmin=mod.prec
 realisable=MPCCmod.viol_comp(mod,xk)<=mod.prec && MPCCmod.viol_cons(mod,xk)<=mod.prec
 solved=true

srelax_xtab=collect(xk[1:n])

#Major Loop
j=0
println("j :",j," xk :",xk[1:n]," f(xk) :",mod.mp.f(xk)," rho :","-"," k :","-")
 while (t+r+s)>pmin && !(realisable) && solved

 # resolution du sous-problème
 if name_relax=="ALAS"
  xk,solved,s_xtab,rho = solve_subproblem(mod,solve_subproblem_alas,r,s,t,name_relax)
 else
  xk,solved = solve_subproblem(mod,solve_subproblem_ipopt,r,s,t,name_relax)
 end

 mod=MPCCmod.addInitialPoint(mod,xk[1:n]) #met à jour le MPCC avec le nouveau point

 r=r*sigma_r
 s=s*sigma_s
 t=t*sigma_t
 append!(srelax_xtab,collect(xk[1:n]))
 realisable=MPCCmod.viol_comp(mod,xk)<=mod.prec && MPCCmod.viol_cons(mod,xk)<=mod.prec
 j+=1
println("j :",j," xk :",xk[1:n]," f(xk) :",mod.mp.f(xk)," rho :","-"," k :","-")
 end

 # output
 return xk, mod.mp.f(xk), stat, srelax_xtab
end

"""
Redefinitions de la fonction solve pour MPCC :
"""
function solve(mod::MPCCmod.MPCC,r0::Float64,sigma_r::Float64,s0::Float64,sigma_s::Float64,t0::Float64,sigma_t::Float64)
 solve(mod,r0,sigma_r,s0,sigma_s,t0,sigma_t,"ALAS")
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
function solve_subproblem(mod::MPCCmod.MPCC, func::Function,r::Float64,s::Float64,t::Float64,name_relax::AbstractString)
 return func(mod,r,s,t,name_relax) #renvoie xk,stat
end

"""
Methode pour résoudre le sous-problème relaxé :
"""
function solve_subproblem_ipopt(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64,name_relax::AbstractString)
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
"""
function solve_subproblem_alas(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64,name_relax::AbstractString)
 solved=true

 prec=max(s,t,r)
 alas = ALASMPCCmod.ALASMPCC(mod,r,s,t,prec)
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
