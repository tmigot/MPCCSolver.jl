module MPCCmod

using NLPModels

import NLPModels.AbstractNLPModel

"""
Definit le type MPCC :
min_x f(x)
l <= x <= u
lb <= c(x) <= ub
0 <= G(x) _|_ H(x) >= 0

liste des constructeurs :
MPCC(f::Function,x0::Vector,
     G::NLPModels.AbstractNLPModel,H::NLPModels.AbstractNLPModel,
     ncc::Int64,
     lvar::Vector,uvar::Vector,
     c::Function,lcon::Vector,ucon::Vector)
MPCC(mp::NLPModels.AbstractNLPModel,
     G::NLPModels.AbstractNLPModel,H::NLPModels.AbstractNLPModel;ncc::Float64=NaN)

liste des accesseurs :


liste des fonctions :

"""

type MPCC #<: AbstractNLPModel

 mp :: AbstractNLPModel
 G  :: AbstractNLPModel
 H  :: AbstractNLPModel

 ncc :: Int64 #nb of complementarity constraints
 nbc :: Int64 #nb of constraints (non-linéaire+bornes+complémentarité)
 n   :: Int64 #dimension of the problem

end

############################################################################

#Constructeurs supplémentaires :

############################################################################

function MPCC(f    :: Function,
              x0   :: Vector,
              G    :: AbstractNLPModel,
              H    :: AbstractNLPModel,
              ncc  :: Int64,
              lvar :: Vector,
              uvar :: Vector,
              c    :: Function,
              lcon :: Vector,
              ucon :: Vector)

 mp = ADNLPModel(f, x0, lvar = lvar, uvar = uvar, c = c, lcon = lcon, ucon = ucon)

 nbc = length(mp.meta.lvar) + length(mp.meta.uvar) + length(mp.meta.lcon) + length(mp.meta.ucon) + 2*ncc
 #et mp.meta.mp.ncon ? - pourquoi length ?

 n = length(mp.meta.x0)

 return MPCC(mp, G, H, ncc, nbc, n)
end

function MPCC(mp  :: AbstractNLPModel;
              G   :: AbstractNLPModel = SimpleNLPModel(x->0, [0.0]),
              H   :: AbstractNLPModel = SimpleNLPModel(x->0, [0.0]),
              ncc :: Int64=-1)

 ncc = ncc == -1 ? G.meta.ncon : ncc
 n = length(mp.meta.x0)
 nbc = 2*(n + mp.meta.ncon + ncc)

 return MPCC(mp, G, H, ncc, nbc, n)
end

############################################################################

# Getteur

############################################################################

function obj(mod :: MPCC, x :: Vector)
 return NLPModels.obj(mod.mp, x)
end

function grad(mod :: MPCC, x :: Vector)
 return NLPModels.grad(mod.mp, x)
end

function grad!(mod :: MPCC, x :: Vector, gx :: Vector)
 return NLPModels.grad(mod.mp, x, gx)
end

function hess(mod :: MPCC, x :: Vector)
 return NLPModels.hess(mod.mp, x)
end

function cons(mod :: MPCC, x :: Vector)

 n = mod.n
 x = length(x) == n ? x : x[1:n]

 feas_x = vcat(max.(mod.mp.meta.lvar-x, 0), max.(x-mod.mp.meta.uvar, 0))

 if mod.mp.meta.ncon !=0

  c = NLPModels.cons(mod.mp, x)
  feas_c = vcat(max.(mod.mp.meta.lcon-c, 0), max.(c-mod.mp.meta.ucon, 0))

 else

  feas_c = []

 end

 G=NLPModels.cons(mod.G, x)
 H=NLPModels.cons(mod.H, x)

 if ncc != 0
  feas_cp = vcat(max.(mod.G.meta.lvar-G, 0), max.(mod.H.meta.lvar-H, 0))
  feas_cc = max.(G.*H, 0)
 else
  feas_cp = []
  feas_cc = []
 end

 return vcat(feas_x, feas_c, feas_cp, feas_cc)
end

function cons_mp(mod :: MPCC, x :: Vector)

 n = mod.n
 x = length(x) == n ? x : x[1:n]

 feas_x = vcat(max.(mod.mp.meta.lvar-x, 0), max.(x-mod.mp.meta.uvar, 0))

 if mod.mp.meta.ncon !=0

  c = NLPModels.cons(mod.mp, x)
  feas_c = vcat(max.(mod.mp.meta.lcon-c, 0), max.(c-mod.mp.meta.ucon, 0))

 else

  feas_c = []

 end

 return vcat(feas_x, feas_c)
end

function cons_nl(mod :: MPCC, x :: Vector)
 return NLPModels.cons(mod.mp, x)
end

function consG(mod :: MPCC, x :: Vector)
 return NLPModels.cons(mod.G, x)
end

function consH(mod :: MPCC, x :: Vector)
 return NLPModels.cons(mod.H, x)
end

function jacG(mod :: MPCC, x :: Vector)
 return NLPModels.jac(mod.G, x)
end

function jacH(mod :: MPCC, x :: Vector)
 return NLPModels.jac(mod.H, x)
end

function jac_nl(mod :: MPCC, x :: Vector)
 return NLPModels.jac(mod.mp, x)
end

"""
Jacobienne des contraintes actives à precmpcc près
"""

function jac_actif(mod :: MPCC, x :: Vector, prec :: Float64)

  n = mod.n

  Il = find(z->z<=prec,abs.(x-mod.mp.meta.lvar))
  Iu = find(z->z<=prec,abs.(x-mod.mp.meta.uvar))
  jl = zeros(n);jl[Il]=1.0;Jl=diagm(jl);
  ju = zeros(n);ju[Iu]=1.0;Ju=diagm(ju);

  IG=[];IH=[];Ig=[];Ih=[];

 if mod.mp.meta.ncon+mod.ncc ==0

  A=[]

 else
  c=NLPModels.cons(mod.mp,x)
  Ig=find(z->z<=prec,abs.(c-mod.mp.meta.lcon))
  Ih=find(z->z<=prec,abs.(c-mod.mp.meta.ucon))
  J = jac_nl(mod, x)
  Jg=J[Ig,1:n]
  Jh=J[Ih,1:n]

  if mod.ncc>0
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
 if mod.ncc>0
  G=NLPModels.cons(mod.G,x)
  H=NLPModels.cons(mod.H,x)
  return [G-yg;H-yh;max.(mod.mp.meta.lvar-x,0);max.(x-mod.mp.meta.uvar,0);max.(mod.mp.meta.lcon-c,0);max.(c-mod.mp.meta.ucon,0)]
 else
  return [yg;yh;max.(mod.mp.meta.lvar-x,0);max.(x-mod.mp.meta.uvar,0);max.(mod.mp.meta.lcon-c,0);max.(c-mod.mp.meta.ucon,0)]
 end

end

function viol_contrainte(mod::MPCC,x::Vector) #x de taille n+2ncc
 n=length(mod.mp.meta.x0)

 return viol_contrainte(mod,x[1:n],x[n+1:n+mod.ncc],x[n+mod.ncc+1:n+2*mod.ncc])
end

"""
Donne la norme de la violation de la complémentarité min(G,H)
"""
function viol_comp(mod::MPCC,x::Vector)

 n=mod.n
 x=length(x)==n?x:x[1:n]

 G=NLPModels.cons(mod.G,x)
 H=NLPModels.cons(mod.H,x)

 return mod.ncc>0?G.*H./(G+H+1):[] #et les contraintes de positivité ?
end

function viol_cons(mod   :: MPCC,
                   x     :: Vector)

 n=mod.n

 x=length(x)==n?x:x[1:n]

 feas_x=vcat(max.(mod.mp.meta.lvar-x,0),max.(x-mod.mp.meta.uvar,0))

 if mod.mp.meta.ncon !=0

  c=cons(mod.mp,x)
  feas_c=vcat(max.(mod.mp.meta.lcon-c,0),max.(c-mod.mp.meta.ucon,0))
 else
  feas_c = []
 end

 return vcat(feas_x,feas_c)
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
