module MPCCmod

using NLPModels #devrait disparaitre

import Relaxation

"""
Definit le type MPCC :
min_x f(x)
l <= x <= u
lb <= c(x) <= ub
0 <= G(x) _|_ H(x) >= 0

liste des constructeurs :
function MPCC(f::Function,x0::Vector,G::Function,H::Function,z0::Vector,nb_comp::Int64;lvar::Vector,uvar::Vector,c::Function,y0::Vector,lcon::Vector,ucon::Vector,name::String)
MPCC(mp::NLPModels.AbstractNLPModel,G::Function,H::Function,z0::Vector,nb_comp::Int64)

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
# - donner G et H à travers un NLPModel (voir Biniveau par la même occasion)
# - MPCCtoRelaxNLP : bug à corriger
#Minor :
# - Comment on modifie le point initial d'un NLPModels ?
# - ne pas utiliser NLPModels

type MPCC
 mp::NLPModels.AbstractNLPModel
 G::Function #the left-side of the complementarity constraint
 H::Function #the right-side of the complementarity constraint
 nb_comp::Int64
 #paramètres pour la résolution :
 prec::Float64 #precision à 0
end

#Constructeurs supplémentaires :
function MPCC(f::Function,x0::Vector,G::Function,H::Function,nb_comp::Int64,lvar::Vector,uvar::Vector,c::Function,y0::Vector,lcon::Vector,ucon::Vector)
 mp=ADNLPModel(f, x0, lvar=lvar, uvar=uvar, y0=y0, c=c, lcon=lcon, ucon=ucon)
 return MPCC(mp,G,H,nb_comp,1e-3)
end

function MPCC(f::Function,x0::Vector,G::Function,H::Function,nb_comp::Int64,lvar::Vector,uvar::Vector,c::Function,y0::Vector,lcon::Vector,ucon::Vector,prec::Float64)
 mp=ADNLPModel(f, x0, lvar=lvar, uvar=uvar, y0=y0, c=c, lcon=lcon, ucon=ucon)
 return MPCC(mp,G,H,nb_comp,prec)
end

function MPCC(mp::NLPModels.AbstractNLPModel,G::Function,H::Function,nb_comp::Int64)
 return MPCC(mp,G,H,nb_comp,1e-3)
end

function addInitialPoint(mod::MPCC,x0::Vector)
 mod.mp=ADNLPModel(mod.mp.f, x0, lvar=mod.mp.meta.lvar, uvar=mod.mp.meta.uvar, y0=mod.mp.meta.y0, c=mod.mp.c, lcon=mod.mp.meta.lcon, ucon=mod.mp.meta.ucon)
 return mod
end

"""
Donne la norme 2 de la violation des contraintes avec slack
"""
function viol_contrainte_norm(mod::MPCCmod.MPCC,x::Vector,yg::Vector,yh::Vector)
 return norm(mod.G(x)-yg)^2+norm(mod.H(x)-yh)^2+norm(max(mod.mp.meta.lvar-x,0))^2+norm(max(x-mod.mp.meta.uvar,0))^2+norm(max(mod.mp.meta.lcon-mod.mp.c(x),0))^2+norm(max(mod.mp.c(x)-mod.mp.meta.ucon,0))^2
end

function viol_contrainte_norm(mod::MPCCmod.MPCC,x::Vector) #x de taille n+2nb_comp
 n=length(mod.mp.meta.x0)
 return viol_contrainte_norm(mod,x[1:n],x[n+1:n+mod.nb_comp],x[n+mod.nb_comp+1:n+2*mod.nb_comp])
end

"""
Donne le vecteur de violation des contraintes dans l'ordre : G(x)-yg ; H(x)-yh ; lvar<=x ; x<=uvar ; lvar<=c(x) ; c(x)<=uvar
"""
function viol_contrainte(mod::MPCCmod.MPCC,x::Vector,yg::Vector,yh::Vector)
 return [mod.G(x)-yg;mod.H(x)-yh;max(mod.mp.meta.lvar-x,0);max(x-mod.mp.meta.uvar,0);max(mod.mp.meta.lcon-mod.mp.c(x),0);max(mod.mp.c(x)-mod.mp.meta.ucon,0)]
end

function viol_contrainte(mod::MPCCmod.MPCC,x::Vector) #x de taille n+2nb_comp
 n=length(mod.mp.meta.x0)
 return viol_contrainte(mod,x[1:n],x[n+1:n+mod.nb_comp],x[n+mod.nb_comp+1:n+2*mod.nb_comp])
end

"""
Donne la norme Inf de la violation de la complémentarité min(G,H)
"""
function viol_comp(mod::MPCCmod.MPCC,x::Vector)
 return norm(mod.G(x).*mod.H(x),Inf)
end

"""
Donne la norme Inf de la violation des contraintes \"classiques\"
"""
function viol_cons(mod::MPCCmod.MPCC,x::Vector)
 return max(maximum(mod.mp.meta.lcon-mod.mp.c(x)),maximum(mod.mp.c(x)-mod.mp.meta.ucon))
end

"""
MPCCtoRelaxNLP(mod::MPCC, t::Float64)
mod : MPCC
return : le MPCC en version NL pour un t donné
"""
function MPCCtoRelaxNLP(mod::MPCC, r::Float64, s::Float64, t::Float64, relax::AbstractString)

 #concatène les contraintes de complémentarité + positivité :
 nl_constraint(x)=emptyfunc
 if relax=="SS"
  nl_constraint(x)=[mod.mp.c(x);mod.G(x).*mod.H(x)-t;mod.G(x);mod.H(x)]
 elseif relax=="KDB" #(G(x)-s)(H(x)-s)<=0, G(x)>=-r, H(x)>=-r
  nl_constraint(x)=[mod.mp.c(x);(mod.G(x)-s).*(mod.H(x)-s);mod.G(x)+r;mod.H(x)+r]
 elseif relax=="KS" #si G(x)-s+H(x)-s>=0 ? (G(x)-s)(H(x)-s)<=0 : -1/2*((G(x)-s)^2+(H(x)-s)^2), G(x)>=0, H(x)>=0
  KS(x)= mod.G(x)-s+mod.H(x)-s>=0 ? (mod.G(x)-s).*(mod.H(x)-s) : -0.5*((mod.G(x)-s).^2+(mod.H(x)-s).^2)
  nl_constraint(x)=[mod.mp.c(x);KS(x);mod.G(x);mod.H(x)]
 elseif relax=="Butterfly"
# On devrait appeler Relaxation et pas Thetamod
#  FG(x)=mod.G(x)-s-t*Thetamod.theta(mod.H(x)-s,r) Bug à corriger
#  FH(x)=mod.H(x)-s-t*Thetamod.theta(mod.G(x)-s,r) Bug à corriger
  FG(x)=mod.G(x)-s-t*(mod.H(x)-s)
  FH(x)=mod.H(x)-s-t*(mod.G(x)-s)
  B(x)= FG(x)+FH(x)>=0 ? FG(x).*FH(x) : -0.5*(FG(x).^2+FH(x).^2)
  nl_constraint(x)=[mod.mp.c(x);B(x);mod.G(x)+r;mod.H(x)+r]
 else
  println("No matching relaxation name. Default : No relaxation. Try : SS, KDB, KS or Butterfly")
  nl_constraint(x)=[mod.mp.c(x);mod.G(x).*mod.H(x);mod.G(x);mod.H(x)]
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
