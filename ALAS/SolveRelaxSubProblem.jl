"""
Package de fonctions pour définir la précision
sur la réalisabilité dual pendant la pénalisation
'sans contraintes'.


liste des fonctions :
SolveSubproblemAlas(mod::MPCCmod.MPCC,
                    r::Float64,s::Float64,t::Float64,
                    rho::Vector,name_relax::AbstractString)
SolveSubproblemIpopt(mod::MPCCmod.MPCC,
                              r::Float64,s::Float64,t::Float64,
                              rho::Vector,name_relax::AbstractString)

MPCCtoRelaxNLP(mod::MPCC, r::Float64, s::Float64, t::Float64,
              relax::AbstractString)

"""

# TO DO List
#Major :
# - MPCCtoRelaxNLP : bug à corriger

module SolveRelaxSubProblem

using MPCCmod
using ALASMPCCmod
using AlgoSetmod
using ParamSetmod

using Ipopt
using MathProgBase
using NLPModels #Pour le MPCCtoRelaxNLP

"""
Methode pour résoudre le sous-problème relaxé :
"""
function SolveSubproblemAlas(mod::MPCCmod.MPCC,
                             r::Float64,s::Float64,t::Float64,
                             rho::Vector,name_relax::AbstractString,paramset::ParamSetmod.ParamSet,algoset::AlgoSetmod.AlgoSet,x0::Vector)
 solved=true

 prec=paramset.prec_oracle(r,s,t,paramset.precmpcc) #Il faut réflechir un peu plus sur des alternatives

 alas = ALASMPCCmod.ALASMPCC(mod,r,s,t,prec,rho,paramset,algoset,x0)

 xk,stat,rho,oa = ALASMPCCmod.solvePAS(alas) #lg,lh,lphi,s_xtab dans le output

 if stat != 0
  println("Error solve_subproblem_alas : ",stat)
  solved=false
 end

 return xk,solved,rho,oa
end

"""
Methode pour résoudre le sous-problème relaxé :
"""
function SolveSubproblemIpOpt(mod::MPCCmod.MPCC,
                              r::Float64,s::Float64,t::Float64,
                              rho::Vector,name_relax::AbstractString)
 solved=true
 nlp_relax = MPCCtoRelaxNLP(mod,r,s,t,name_relax)
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


###
#
# !!! Ne marche pas comme ça !!!
#
###
"""
MPCCtoRelaxNLP(mod::MPCC, t::Float64)
mod : MPCC
return : le MPCC en version NL pour un t donné
"""
function MPCCtoRelaxNLP(mod::MPCCmod.MPCC, r::Float64, s::Float64, t::Float64, relax::AbstractString)

 G(x)=NLPModels.cons(mod.G,x)
 H(x)=NLPModels.cons(mod.H,x)

 #concatène les contraintes de complémentarité + positivité :
 tc(x)=(c=zeros(mod.mp.meta.ncon);MathProgBase.eval_g(mod.mp.mpmodel.eval, c, x);c)

 if relax=="SS"
  nl_constraint(x)=[G(x).*H(x)-t;G(x);H(x)]
 elseif relax=="KDB" #(G(x)-s)(H(x)-s)<=0, G(x)>=-r, H(x)>=-r
  nl_constraint(x)=[(G(x)-s).*(H(x)-s);G(x)+r;H(x)+r]
 elseif relax=="KS" #si G(x)-s+H(x)-s>=0 ? (G(x)-s)(H(x)-s)<=0 : -1/2*((G(x)-s)^2+(H(x)-s)^2), G(x)>=0, H(x)>=0
  KS(x)= G(x)-s+H(x)-s>=0 ? (G(x)-s).*(H(x)-s) : -0.5*((G(x)-s).^2+(H(x)-s).^2)
  nl_constraint(x)=[KS(x);G(x);H(x)]
 elseif relax=="Butterfly"
# On devrait appeler Relaxation et pas Thetamod
#  FG(x)=mod.G(x)-s-t*Thetamod.theta(mod.H(x)-s,r) Bug à corriger
#  FH(x)=mod.H(x)-s-t*Thetamod.theta(mod.G(x)-s,r) Bug à corriger
  FG(x)=G(x)-s-t*(H(x)-s)
  FH(x)=H(x)-s-t*(G(x)-s)
  B(x)= FG(x)+FH(x)>=0 ? FG(x).*FH(x) : -0.5*(FG(x).^2+FH(x).^2)
  nl_constraint(x)=[B(x);G(x)+r;H(x)+r]
 else
  println("No matching relaxation name. Default : No relaxation. Try : SS, KDB, KS or Butterfly")
  nl_constraint(x)=[G(x).*H(x);G(x);H(x)]
 end

 lcon=[mod.mp.meta.lcon;-Inf*ones(mod.nb_comp);zeros(mod.nb_comp*2)]
 ucon=[mod.mp.meta.ucon;zeros(mod.nb_comp);Inf*ones(mod.nb_comp*2)]
 y0=[mod.mp.meta.y0;zeros(3*mod.nb_comp)]

 nlc(x)=[tc(x);nl_constraint(x)]
 f(x)=MathProgBase.eval_f(mod.mp.mpmodel.eval, x)

 #appel au constructeur NLP que l'on souhaite utiliser.
 nlp = ADNLPModel(f, mod.mp.meta.x0, lvar=mod.mp.meta.lvar, uvar=mod.mp.meta.uvar, y0=y0, c=nlc, lcon=lcon, ucon=ucon)

 return nlp
end


#end of module
end
