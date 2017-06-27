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
using OutputRelaxationmod

using Ipopt
using NLPModels
using MathProgBase
#using PyPlot ??

"""
Methode de relaxation pour resoudre :
(r,s,t) devrait apparaitre dans des choix de stratégie algorithmique...
"""
function solve(mod::MPCCmod.MPCC,r0::Float64=0.1,sigma_r::Float64=0.01,s0::Float64=0.1,sigma_s::Float64=0.01,t0::Float64=0.1,sigma_t::Float64=0.01;name_relax::AbstractString="ALAS")
#initialization
 t=t0;r=r0;s=s0;

 rho=mod.paramset.rho_init
 n=length(mod.mp.meta.x0)
 x0=mod.mp.meta.x0
 xk=x0
 pmin=mod.paramset.paramin

 real=MPCCmod.viol_contrainte_norm(mod,xk)
 realisable=real<=mod.paramset.precmpcc
 solved=true
 param=true
 or=OutputRelaxationmod.OutputRelaxation(xk,real, NLPModels.obj(mod.mp,xk))
 optimal=stationary_check(mod,x0) #reste à checker le signe des multiplicateurs

#Major Loop
j=0
 while param && !(realisable && solved && optimal) 

  # resolution du sous-problème
  if name_relax=="ALAS"
   xk,solved,rho,output = solve_subproblem(mod,solve_subproblem_alas,r,s,t,rho,name_relax)
  else
   xk,solved,rho,output = solve_subproblem(mod,solve_subproblem_ipopt,r,s,t,rho,name_relax)
  end

  real=MPCCmod.viol_contrainte_norm(mod,xk[1:n])
  or=OutputRelaxationmod.UpdateOR(or,xk[1:n],0,r,s,t,mod.paramset.prec_oracle(r,s,t,mod.paramset.precmpcc),real,output,NLPModels.obj(mod.mp,xk))

  mod=MPCCmod.addInitialPoint(mod,xk[1:n]) #met à jour le MPCC avec le nouveau point

  r=r*sigma_r
  s=s*sigma_s
  t=t*sigma_t

  solved=true in isnan(xk)?false:solved
  realisable=real<=mod.paramset.precmpcc
  optimal=stationary_check(mod,xk[1:n])
  param=(t+r+s)>pmin

  j+=1
 end

#Traitement final :
OutputRelaxationmod.Print(or,n,mod.paramset.verbose)

realisable || print_with_color(:green,"Infeasible solution: (comp,cons)=($(MPCCmod.viol_comp(mod,xk)),$(MPCCmod.viol_cons(mod,xk)))\n" )
solved || print_with_color(:green,"Subproblem failure. NaN in the solution ? $(true in isnan(xk)). Stationary ? $(realisable && optimal)\n")
param || realisable || print_with_color(:green,"Parameters too small\n")
solved && realisable && print_with_color(:green,"Success\n")

 mod=MPCCmod.addInitialPoint(mod,x0[1:n]) #remet le point initial du MPCC

 # output
 return xk, NLPModels.obj(mod.mp,xk), or
end

"""
Méthode de relaxation avec penalisation des paramètres
"""
function solve(mod::MPCCmod.MPCC)
 if mod.nb_comp==0
  return solve(mod,0.1,0.0,0.1,0.0,0.1,0.0,name_relax="ALAS")
 else
  return solve(mod,0.1,0.01,0.1,0.01,0.1,0.01,name_relax="ALAS")
 end
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
 model_relax = NLPModels.NLPtoMPB(nlp_relax, IpoptSolver(print_level=0,tol=mod.paramset.precmpcc))
 MathProgBase.optimize!(model_relax)

 if MathProgBase.status(model_relax) == :Optimal
  xk = MathProgBase.getsolution(model_relax)
  solved=false
 else
 end

 return xk,solved,rho,output
end

"""
Methode pour résoudre le sous-problème relaxé :

note : le choix de la stratégie sur rho devrait être décidé dans mod
"""
function solve_subproblem_alas(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64,rho::Vector,name_relax::AbstractString)
 solved=true

 prec=mod.paramset.prec_oracle(r,s,t,mod.paramset.precmpcc) #Il faut réflechir un peu plus sur des alternatives

 if mod.paramset.rho_restart
  alas = ALASMPCCmod.ALASMPCC(mod,r,s,t,prec) #recommence rho à chaque itération
 else
  alas = ALASMPCCmod.ALASMPCC(mod,r,s,t,prec,rho)
 end

 xk,stat,rho,oa = ALASMPCCmod.solvePAS(alas) #lg,lh,lphi,s_xtab dans le output

 if stat != 0
  println("Error solve_subproblem_alas : ",stat)
  solved=false
 end

 return xk,solved,rho,oa
end

function stationary_check(mod::MPCCmod.MPCC,x::Vector)

 b=-NLPModels.grad(mod.mp,x)

 if mod.mp.meta.ncon+mod.nb_comp ==0
  optimal=norm(b,Inf)<=mod.paramset.precmpcc
 else
  if mod.nb_comp>0
   A=[NLPModels.jac(mod.mp,x); NLPModels.jac(mod.G,x); NLPModels.jac(mod.H,x) ]'
  else
   A=NLPModels.jac(mod.mp,x)'
  end
  l=pinv(full(A))*b #pinv not defined for sparse matrix
  optimal=maximum(max(A*l-b,0))<=mod.paramset.precmpcc

#Checker les signes des multiplicateurs est virtuellement impossible...

 end

 return optimal
end

#end of module
end
