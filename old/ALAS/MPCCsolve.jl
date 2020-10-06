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

 real=MPCCmod.viol_contrainte_norm(mod,xk);feas=NaN #test
 realisable=real<=mod.paramset.precmpcc
 solved=true
 param=true
 f=NLPModels.obj(mod.mp,xk)
 or=OutputRelaxationmod.OutputRelaxation(xk,real, f)
 if realisable
  optimal=stationary_check(mod,x0) #reste à checker le signe des multiplicateurs
 else
  optimal=false
 end

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
  f=NLPModels.obj(mod.mp,xk[1:n])

  or=OutputRelaxationmod.UpdateOR(or,xk[1:n],0,r,s,t,mod.paramset.prec_oracle(r,s,t,mod.paramset.precmpcc),real,output,f)

  mod=MPCCmod.addInitialPoint(mod,xk[1:n]) #met à jour le MPCC avec le nouveau point

  r=r*sigma_r
  s=s*sigma_s
  t=t*sigma_t

  solved=true in isnan.(xk)?false:solved
  realisable=real<=mod.paramset.precmpcc

  optimal=!isnan(f) && !(true in isnan.(xk)) && stationary_check(mod,xk[1:n])
  param=(t+r+s)>pmin

  j+=1
 end

if mod.paramset.verbose != 0.0
 realisable || print_with_color(:green,"Infeasible solution: (comp,cons)=($(MPCCmod.viol_comp(mod,xk)),$(MPCCmod.viol_cons(mod,xk)))\n" )
 solved || print_with_color(:green,"Subproblem failure. NaN in the solution ? $(true in isnan(xk)). Stationary ? $(realisable && optimal)\n")
 param || realisable || print_with_color(:green,"Parameters too small\n") 
 solved && realisable && optimal && print_with_color(:green,"Success\n")
end

if solved && optimal && realisable
 or=OutputRelaxationmod.UpdateFinalOR(or,"Success")
elseif optimal && realisable
 or=OutputRelaxationmod.UpdateFinalOR(or,"Success (with sub-pb failure)")
elseif !realisable
 or=OutputRelaxationmod.UpdateFinalOR(or,"Infeasible")
elseif realisable && !optimal
 or=OutputRelaxationmod.UpdateFinalOR(or,"Feasible, but not optimal")
else
 or=OutputRelaxationmod.UpdateFinalOR(or,"autres")
end

#Traitement final :
OutputRelaxationmod.Print(or,n,mod.paramset.verbose)

 mod=MPCCmod.addInitialPoint(mod,x0[1:n]) #remet le point initial du MPCC
 nb_eval=[mod.mp.counters.neval_obj,mod.mp.counters.neval_cons,
          mod.mp.counters.neval_grad,mod.mp.counters.neval_hess,
          mod.G.counters.neval_cons,mod.G.counters.neval_jac,
          mod.H.counters.neval_cons,mod.H.counters.neval_jac]

 # output
 return xk, f, or, nb_eval
end

"""
Méthode de relaxation avec penalisation des paramètres
"""
function solve(mod::MPCCmod.MPCC)
 if mod.nb_comp==0
  return solve(mod,0.0,0.0,0.0,0.0,0.0,0.0,name_relax="ALAS")
 else
  return solve(mod,0.1,0.1,0.1,0.1,0.1,0.1,name_relax="ALAS")
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

 alas = ALASMPCCmod.ALASMPCC(mod,r,s,t,prec,rho)

 xk,stat,rho,oa = ALASMPCCmod.solvePAS(alas) #lg,lh,lphi,s_xtab dans le output

 if stat != 0
  println("Error solve_subproblem_alas : ",stat)
  solved=false
 end

 return xk,solved,rho,oa
end

function stationary_check(mod::MPCCmod.MPCC,x::Vector)
 n=length(x)
 prec=mod.paramset.precmpcc
 b=-NLPModels.grad(mod.mp,x)

  Il=find(z->norm(z-mod.mp.meta.lvar,Inf)<=prec,x)
  Iu=find(z->norm(z-mod.mp.meta.uvar,Inf)<=prec,x)
  jl=zeros(n);jl[Il]=1.0;Jl=diagm(jl);
  ju=zeros(n);jl[Iu]=1.0;Ju=diagm(ju);

 if mod.mp.meta.ncon+mod.nb_comp ==0

  optimal=norm(b,Inf)<=mod.paramset.precmpcc

 else
  c=cons(mod.mp,x)
  Ig=find(z->norm(z-mod.mp.meta.lcon,Inf)<=prec,c)
  Ih=find(z->norm(z-mod.mp.meta.ucon,Inf)<=prec,c)
  Jg=NLPModels.jac(mod.mp,x)[Ig,1:n]
  Jh=NLPModels.jac(mod.mp,x)[Ih,1:n]

  if mod.nb_comp>0
   IG=find(z->norm(z-mod.G.meta.lcon,Inf)<=prec,NLPModels.cons(mod.G,x))
   IH=find(z->norm(z-mod.H.meta.lcon,Inf)<=prec,NLPModels.cons(mod.H,x))
   A=[Jl;Ju;-Jg;Jh; -NLPModels.jac(mod.G,x)[IG,1:n]; -NLPModels.jac(mod.H,x)[IH,1:n] ]'
  else
   A=[Jl;Ju;-Jg;Jh]'
  end

  if !(true in isnan.(A) || true in isnan.(b))
   l=pinv(full(A))*b #pinv not defined for sparse matrix
   optimal=0.5*norm(A*l-b,2)^2<=mod.paramset.precmpcc
  else
   @printf("Evaluation error: NaN in the derivative")
   optimal=false
  end
#Checker les signes des multiplicateurs est virtuellement impossible...

 end

 return optimal
end

#end of module
end
