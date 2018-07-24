module MPCCmod

using NLPModels

"""
Definit le type MPCC :
min_x f(x)
l <= x <= u
lb <= c(x) <= ub
0 <= G(x) _|_ H(x) >= 0

liste des constructeurs :
MPCC(f::Function,x0::Vector,
     G::NLPModels.AbstractNLPModel,H::NLPModels.AbstractNLPModel,
     nb_comp::Int64,
     lvar::Vector,uvar::Vector,
     c::Function,lcon::Vector,ucon::Vector)
MPCC(mp::NLPModels.AbstractNLPModel,
     G::NLPModels.AbstractNLPModel,H::NLPModels.AbstractNLPModel,nb_comp)
MPCC(mp::NLPModels.AbstractNLPModel)
MPCC(mp::NLPModels.AbstractNLPModel,algo::AlgoSetmod.AlgoSet)
MPCC(mp::NLPModels.AbstractNLPModel,
     G::NLPModels.AbstractNLPModel,H::NLPModels.AbstractNLPModel;nb_comp::Float64=NaN)

liste des accesseurs :
addInitialPoint(mod::MPCC,x0::Vector)
obj(mod::MPCC,x::Vector)
grad(mod::MPCC,x::Vector)
jac_actif(mod::MPCC,x::Vector)

liste des fonctions :
viol_contrainte_norm(mod::MPCCmod.MPCC,x::Vector,yg::Vector,yh::Vector)
viol_contrainte_norm(mod::MPCCmod.MPCC,x::Vector)
viol_contrainte(mod::MPCCmod.MPCC,x::Vector,yg::Vector,yh::Vector)
viol_contrainte(mod::MPCCmod.MPCC,x::Vector)
viol_comp(mod::MPCCmod.MPCC,x::Vector)_ _v'

dual_feasibility(mod::MPCC,x::Vector,l::Vector,A::Any)
sign_stationarity_check(mod::MPCC,x::Vector,l::Vector)
sign_stationarity_check(mod::MPCC,x::Vector,l::Vector,
                        Il::Array{Int64,1},Iu::Array{Int64,1},
                        Ig::Array{Int64,1},Ih::Array{Int64,1},
                        IG::Array{Int64,1},IH::Array{Int64,1})
stationary_check(mod::MPCC,x::Vector)
"""

type MPCC

 mp :: NLPModels.AbstractNLPModel
 G  :: NLPModels.AbstractNLPModel
 H  :: NLPModels.AbstractNLPModel

 nb_comp :: Int64 #nb of complementarity constraints
 nbc     :: Int64 #nb of constraints (non-linéaire+bornes+complémentarité)
 n       :: Int64 #dimension of the problem

end

############################################################################

#Constructeurs supplémentaires :

############################################################################

function MPCC(f::Function,
              x0::Vector,
              G::NLPModels.AbstractNLPModel,
              H::NLPModels.AbstractNLPModel,
              nb_comp::Int64,
              lvar::Vector,
              uvar::Vector,
              c::Function,
              lcon::Vector,
              ucon::Vector)

 mp=ADNLPModel(f, x0, lvar=lvar, uvar=uvar, c=c, lcon=lcon, ucon=ucon)

 nbc=length(mp.meta.lvar)+length(mp.meta.uvar)+length(mp.meta.lcon)+length(mp.meta.ucon)+2*nb_comp
 n=length(mp.meta.x0)

 return MPCC(mp,G,H,nb_comp,nbc,n)
end

function MPCC(mp::NLPModels.AbstractNLPModel,
              G::NLPModels.AbstractNLPModel,
              H::NLPModels.AbstractNLPModel,
              nb_comp::Int64)

 nbc=length(mp.meta.lvar)+length(mp.meta.uvar)+length(mp.meta.lcon)+length(mp.meta.ucon)+2*nb_comp
 n=length(mp.meta.x0)
 
 return MPCC(mp,G,H,nb_comp,nbc,n)
end

function MPCC(mp::NLPModels.AbstractNLPModel)

 #le plus "petit" SimpleNLPModel
 G=SimpleNLPModel(x->0, [0.0])
 H=SimpleNLPModel(x->0, [0.0])

 return MPCC(mp,G,H,nb_comp=0)
end

function MPCC(mp::NLPModels.AbstractNLPModel,
              G::NLPModels.AbstractNLPModel,
              H::NLPModels.AbstractNLPModel;
              nb_comp::Float64=NaN) #erreur Int64

 nb_comp=isnan(nb_comp)?length(NLPModels.cons(G,mp.meta.x0)):nb_comp
 nbc=length(mp.meta.lvar)+length(mp.meta.uvar)+length(mp.meta.lcon)+length(mp.meta.ucon)+2*nb_comp
 n=length(mp.meta.x0)

 return MPCC(mp,G,H,nb_comp,nbc,n)
end

############################################################################

# Getteur

############################################################################

function obj(mod::MPCC,x::Vector)
 return NLPModels.obj(mod.mp,x)
end
"""
gradient de la fonction objectif
"""
function grad(mod::MPCC,x::Vector)
 return NLPModels.grad(mod.mp,x)
end

function hess(mod::MPCC,x::Vector)
 return NLPModels.hess(mod.mp,x)
end

"""
Jacobienne des contraintes actives à precmpcc près
"""

function jac_actif(mod::MPCC,x::Vector,prec)

  n=mod.n

  Il=find(z->z<=prec,abs.(x-mod.mp.meta.lvar))
  Iu=find(z->z<=prec,abs.(x-mod.mp.meta.uvar))
  jl=zeros(n);jl[Il]=1.0;Jl=diagm(jl);
  ju=zeros(n);ju[Iu]=1.0;Ju=diagm(ju);

  IG=[];IH=[];Ig=[];Ih=[];

 if mod.mp.meta.ncon+mod.nb_comp ==0

  A=[]

 else
  c=cons(mod.mp,x)
  Ig=find(z->z<=prec,abs.(c-mod.mp.meta.lcon))
  Ih=find(z->z<=prec,abs.(c-mod.mp.meta.ucon))
  Jg=NLPModels.jac(mod.mp,x)[Ig,1:n]
  Jh=NLPModels.jac(mod.mp,x)[Ih,1:n]

  if mod.nb_comp>0
   IG=find(z->z<=prec,abs.(NLPModels.cons(mod.G,x)-mod.G.meta.lcon))
   IH=find(z->z<=prec,abs.(NLPModels.cons(mod.H,x)-mod.H.meta.lcon))

   A=[Jl;Ju;-Jg;Jh;-NLPModels.jac(mod.G,x)[IG,1:n];-NLPModels.jac(mod.H,x)[IH,1:n]]'
  else
   A=[Jl;Ju;-Jg;Jh]'
  end
 end

 return A, Il,Iu,Ig,Ih,IG,IH
end

"""
Donne le vecteur de violation des contraintes dans l'ordre : G(x)-yg ; H(x)-yh ; lvar<=x ; x<=uvar ; lvar<=c(x) ; c(x)<=uvar
"""
# Pourquoi ça s'appelle pas "cons" ?
function viol_contrainte(mod::MPCC,x::Vector,yg::Vector,yh::Vector)

 c=NLPModels.cons(mod.mp,x)
 if mod.nb_comp>0
  G=NLPModels.cons(mod.G,x)
  H=NLPModels.cons(mod.H,x)
  return [G-yg;H-yh;max.(mod.mp.meta.lvar-x,0);max.(x-mod.mp.meta.uvar,0);max.(mod.mp.meta.lcon-c,0);max.(c-mod.mp.meta.ucon,0)]
 else
  return [yg;yh;max.(mod.mp.meta.lvar-x,0);max.(x-mod.mp.meta.uvar,0);max.(mod.mp.meta.lcon-c,0);max.(c-mod.mp.meta.ucon,0)]
 end

end

function viol_contrainte(mod::MPCC,x::Vector) #x de taille n+2nb_comp
 n=length(mod.mp.meta.x0)

 return viol_contrainte(mod,x[1:n],x[n+1:n+mod.nb_comp],x[n+mod.nb_comp+1:n+2*mod.nb_comp])
end

"""
Donne la norme de la violation de la complémentarité min(G,H)
"""
function viol_comp(mod::MPCC,x::Vector)

 n=mod.n
 x=length(x)==n?x:x[1:n]

 G=NLPModels.cons(mod.G,x)
 H=NLPModels.cons(mod.H,x)

 return mod.nb_comp>0?G.*H./(G+H+1):[]
end

"""
Donne la violation de la réalisabilité dual
"""
function dual_feasibility(mod::MPCC,x::Vector,l::Vector,A::Any) #type général pour matrice ?

 b=grad(mod,x)

 return A*l+b
end

#end du module
end
