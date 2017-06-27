module MPCCmod

using NLPModels

import Relaxation
import ParamSetmod
import AlgoSetmod

"""
Definit le type MPCC :
min_x f(x)
l <= x <= u
lb <= c(x) <= ub
0 <= G(x) _|_ H(x) >= 0

liste des constructeurs :
MPCC(f::Function,x0::Vector,G::Function,H::Function,nb_comp::Int64,lvar::Vector,uvar::Vector,c::Function,y0::Vector,lcon::Vector,ucon::Vector)
MPCC(f::Function,x0::Vector,G::Function,H::Function,nb_comp::Int64,lvar::Vector,uvar::Vector,c::Function,y0::Vector,lcon::Vector,ucon::Vector,prec::Float64)
MPCC(mp::NLPModels.AbstractNLPModel,G::Function,H::Function,nb_comp::Int64)

liste des accesseurs :
addInitialPoint(mod::MPCC,x0::Vector)

liste des fonctions :
viol_contrainte_norm(mod::MPCCmod.MPCC,x::Vector,yg::Vector,yh::Vector)
viol_contrainte_norm(mod::MPCCmod.MPCC,x::Vector)
viol_contrainte(mod::MPCCmod.MPCC,x::Vector,yg::Vector,yh::Vector)
viol_contrainte(mod::MPCCmod.MPCC,x::Vector)
viol_comp(mod::MPCCmod.MPCC,x::Vector)
viol_cons(mod::MPCCmod.MPCC,x::Vector)

MPCCtoRelaxNLP(mod::MPCC, r::Float64, s::Float64, t::Float64, relax::AbstractString)
"""

# TO DO List
#Major :
# - MPCCtoRelaxNLP : bug à corriger
# - appel de la hessienne d'un SimpleNLPModel ? NLPModels.hess(mod.H,x)

type MPCC
 mp::NLPModels.AbstractNLPModel
 G::NLPModels.AbstractNLPModel
 H::NLPModels.AbstractNLPModel
 xj::Vector #itéré courant
 nb_comp::Int64

 algoset::AlgoSetmod.AlgoSet
 paramset::ParamSetmod.ParamSet
end

#Constructeurs supplémentaires :
function MPCC(f::Function,x0::Vector,Gfunc::Function,Hfunc::Function,nb_comp::Int64,lvar::Vector,uvar::Vector,c::Function,y0::Vector,lcon::Vector,ucon::Vector)

#Manque la jacobienne !!
 G=SimpleNLPModel(f, x0, lvar=lvar, uvar=uvar, c=Gfunc, lcon=zeros(nb_comp), ucon=Inf*ones(nb_comp))
 H=SimpleNLPModel(f, x0, lvar=lvar, uvar=uvar, c=Hfunc, lcon=zeros(nb_comp), ucon=Inf*ones(nb_comp))

 mp=ADNLPModel(f, x0, lvar=lvar, uvar=uvar, y0=y0, c=c, lcon=lcon, ucon=ucon)
 nbc=length(mp.meta.lvar)+length(mp.meta.uvar)+length(mp.meta.lcon)+length(mp.meta.ucon)+2*nb_comp

 return MPCC(mp,G,H,x0,nb_comp,AlgoSetmod.AlgoSet(),ParamSetmod.ParamSet(nbc))
end

function MPCC(f::Function,x0::Vector,G::NLPModels.AbstractNLPModel,H::NLPModels.AbstractNLPModel,nb_comp::Int64,lvar::Vector,uvar::Vector,c::Function,lcon::Vector,ucon::Vector)

 mp=ADNLPModel(f, x0, lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon)
 nbc=length(mp.meta.lvar)+length(mp.meta.uvar)+length(mp.meta.lcon)+length(mp.meta.ucon)+2*nb_comp

 return MPCC(mp,G,H,x0,nb_comp,AlgoSetmod.AlgoSet(),ParamSetmod.ParamSet(nbc))
end

function MPCC(mp::NLPModels.AbstractNLPModel,Gfunc::Function,Hfunc::Function,nb_comp::Int64)
 nbc=length(mp.meta.lvar)+length(mp.meta.uvar)+length(mp.meta.lcon)+length(mp.meta.ucon)+2*nb_comp

#Manque la jacobienne !!
 G=SimpleNLPModel(f, x0, lvar=lvar, uvar=uvar, c=Gfunc, lcon=zeros(nb_comp), ucon=Inf*ones(nb_comp))
 H=SimpleNLPModel(f, x0, lvar=lvar, uvar=uvar, c=Hfunc, lcon=zeros(nb_comp), ucon=Inf*ones(nb_comp))

 return MPCC(mp,G,H,mp.meta.x0,nb_comp,AlgoSetmod.AlgoSet(),ParamSetmod.ParamSet(nbc))
end

function MPCC(mp::NLPModels.AbstractNLPModel,G::NLPModels.AbstractNLPModel,H::NLPModels.AbstractNLPModel,nb_comp)
 nbc=length(mp.meta.lvar)+length(mp.meta.uvar)+length(mp.meta.lcon)+length(mp.meta.ucon)+2*nb_comp
 
 return MPCC(mp,G,H,mp.meta.x0,nb_comp,AlgoSetmod.AlgoSet(),ParamSetmod.ParamSet(nbc))
end

function MPCC(mp::NLPModels.AbstractNLPModel)

 G=SimpleNLPModel(x->0, [0.0])
 H=SimpleNLPModel(x->0, [0.0])

 nb_comp=0
 nbc=length(mp.meta.lvar)+length(mp.meta.uvar)+length(mp.meta.lcon)+length(mp.meta.ucon)+2*nb_comp

 return MPCC(mp,G,H,mp.meta.x0,nb_comp,
		AlgoSetmod.AlgoSet(),ParamSetmod.ParamSet(nbc))
end

"""
Accesseur : modifie le point initial
"""

function addInitialPoint(mod::MPCC,x0::Vector)
 mod.xj=x0
 return mod
end

"""
Donne la norme 2 de la violation des contraintes avec slack

note : devrait appeler viol_contrainte
"""
function viol_contrainte_norm(mod::MPCCmod.MPCC,x::Vector,yg::Vector,yh::Vector)

 G(x)=mod.nb_comp!=0?NLPModels.cons(mod.G,x):0
 H(x)=mod.nb_comp!=0?NLPModels.cons(mod.H,x):0
 c(x)=mod.mp.meta.ncon!=0?NLPModels.cons(mod.mp,x):0

 return norm(G(x)-yg)^2+norm(H(x)-yh)^2+norm(max(mod.mp.meta.lvar-x,0))^2+norm(max(x-mod.mp.meta.uvar,0))^2+norm(max(mod.mp.meta.lcon-mod.mp.c(x),0))^2+norm(max(mod.mp.c(x)-mod.mp.meta.ucon,0))^2
end

function viol_contrainte_norm(mod::MPCCmod.MPCC,x::Vector) #x de taille n+2nb_comp

 n=length(mod.mp.meta.x0)
 if length(x)==n
  resul=max(viol_comp(mod,x),viol_cons(mod,x))
 else
  resul=viol_contrainte_norm(mod,x[1:n],x[n+1:n+mod.nb_comp],x[n+mod.nb_comp+1:n+2*mod.nb_comp])
 end
 return resul
end

"""
Donne le vecteur de violation des contraintes dans l'ordre : G(x)-yg ; H(x)-yh ; lvar<=x ; x<=uvar ; lvar<=c(x) ; c(x)<=uvar
"""
function viol_contrainte(mod::MPCCmod.MPCC,x::Vector,yg::Vector,yh::Vector)
 c(z)=NLPModels.cons(mod.mp,z)
 if mod.nb_comp>0
  G(x)=NLPModels.cons(mod.G,x)
  H(x)=NLPModels.cons(mod.H,x)
  return [G(x)-yg;H(x)-yh;max(mod.mp.meta.lvar-x,0);max(x-mod.mp.meta.uvar,0);max(mod.mp.meta.lcon-c(x),0);max(c(x)-mod.mp.meta.ucon,0)]
 else
  return [yg;yh;max(mod.mp.meta.lvar-x,0);max(x-mod.mp.meta.uvar,0);max(mod.mp.meta.lcon-c(x),0);max(c(x)-mod.mp.meta.ucon,0)]
 end
end

function viol_contrainte(mod::MPCCmod.MPCC,x::Vector) #x de taille n+2nb_comp
 n=length(mod.mp.meta.x0)

 return viol_contrainte(mod,x[1:n],x[n+1:n+mod.nb_comp],x[n+mod.nb_comp+1:n+2*mod.nb_comp])
end

"""
Donne la norme Inf de la violation de la complémentarité min(G,H)
"""
function viol_comp(mod::MPCCmod.MPCC,x::Vector)

 return mod.nb_comp>0?norm(NLPModels.cons(mod.G,x).*NLPModels.cons(mod.H,x),Inf):0
end

"""
Donne la norme Inf de la violation des contraintes \"classiques\"
"""
function viol_cons(mod::MPCCmod.MPCC,x::Vector)
 feas=0.0

 if mod.mp.meta.ncon !=0
  c=NLPModels.cons(mod.mp,x)
  feas=max(maximum(mod.mp.meta.lcon-c),maximum(c-mod.mp.meta.ucon))
 end

 return feas
end

"""
MPCCtoRelaxNLP(mod::MPCC, t::Float64)
mod : MPCC
return : le MPCC en version NL pour un t donné
"""
function MPCCtoRelaxNLP(mod::MPCC, r::Float64, s::Float64, t::Float64, relax::AbstractString)
 G(x)=NLPModels.cons(mod.G,x)
 H(x)=NLPModels.cons(mod.H,x)

 #concatène les contraintes de complémentarité + positivité :
 nl_constraint(x)=emptyfunc
 if relax=="SS"
  nl_constraint(x)=[mod.mp.c(x);G(x).*H(x)-t;G(x);H(x)]
 elseif relax=="KDB" #(G(x)-s)(H(x)-s)<=0, G(x)>=-r, H(x)>=-r
  nl_constraint(x)=[mod.mp.c(x);(G(x)-s).*(H(x)-s);G(x)+r;H(x)+r]
 elseif relax=="KS" #si G(x)-s+H(x)-s>=0 ? (G(x)-s)(H(x)-s)<=0 : -1/2*((G(x)-s)^2+(H(x)-s)^2), G(x)>=0, H(x)>=0
  KS(x)= G(x)-s+H(x)-s>=0 ? (G(x)-s).*(H(x)-s) : -0.5*((G(x)-s).^2+(H(x)-s).^2)
  nl_constraint(x)=[mod.mp.c(x);KS(x);G(x);H(x)]
 elseif relax=="Butterfly"
# On devrait appeler Relaxation et pas Thetamod
#  FG(x)=mod.G(x)-s-t*Thetamod.theta(mod.H(x)-s,r) Bug à corriger
#  FH(x)=mod.H(x)-s-t*Thetamod.theta(mod.G(x)-s,r) Bug à corriger
  FG(x)=G(x)-s-t*(H(x)-s)
  FH(x)=H(x)-s-t*(G(x)-s)
  B(x)= FG(x)+FH(x)>=0 ? FG(x).*FH(x) : -0.5*(FG(x).^2+FH(x).^2)
  nl_constraint(x)=[mod.mp.c(x);B(x);G(x)+r;H(x)+r]
 else
  println("No matching relaxation name. Default : No relaxation. Try : SS, KDB, KS or Butterfly")
  nl_constraint(x)=[mod.mp.c(x);G(x).*H(x);G(x);H(x)]
 end

 lcon=[mod.mp.meta.lcon;-Inf*ones(mod.nb_comp);zeros(mod.nb_comp*2)]
 ucon=[mod.mp.meta.ucon;zeros(mod.nb_comp);Inf*ones(mod.nb_comp*2)]
 y0=[mod.mp.meta.y0;zeros(3*mod.nb_comp)]

 #appel au constructeur NLP que l'on souhaite utiliser.
 nlp = ADNLPModel(mod.mp.f, mod.mp.meta.x0, lvar=mod.mp.meta.lvar, uvar=mod.mp.meta.uvar, y0=y0, c=nl_constraint, lcon=lcon, ucon=ucon)

 return nlp
end

#end du module
end
