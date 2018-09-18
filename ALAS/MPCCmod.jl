module MPCCmod

importall NLPModels

import NLPModels: AbstractNLPModel
import MPCCMetamod: MPCCMeta

"""
Definit le type MPCC :
min f(x)
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

type MPCC <: AbstractNLPModel

 mp :: AbstractNLPModel
 G  :: AbstractNLPModel
 H  :: AbstractNLPModel

 meta :: MPCCMeta

 function MPCC(mp  :: AbstractNLPModel;
               G   :: AbstractNLPModel = SimpleNLPModel(x->0, [0.0]),
               H   :: AbstractNLPModel = SimpleNLPModel(x->0, [0.0]),
               ncc :: Int64 = -1,
               x0  :: Vector = mp.meta.x0)

  ncc = ncc == -1 ? G.meta.ncon : ncc 

  if G.meta.ncon != H.meta.ncon throw(error("Incompatible complementarity")) end

  n    = length(x0)
  ncon = mp.meta.ncon

  meta = MPCCMeta(n, x0 = x0, ncc = ncc, lccG = G.meta.lcon, lccH = H.meta.lcon,
                              lvar = mp.meta.lvar, uvar = mp.meta.uvar,
                              ncon = ncon, 
                              lcon = mp.meta.lcon, ucon = mp.meta.ucon)

  return new(mp, G, H, meta)
 end
end

NotImplemented(args...; kwargs...) = throw(NotImplementedError(""))

############################################################################

#Constructeurs supplémentaires :

############################################################################

function MPCC(f    :: Function,
              x0   :: Vector,
              G    :: AbstractNLPModel,
              H    :: AbstractNLPModel,
              lvar :: Vector,
              uvar :: Vector,
              c    :: Function,
              lcon :: Vector,
              ucon :: Vector)

 mp = ADNLPModel(f, x0, lvar = lvar, uvar = uvar, c = c, lcon = lcon, ucon = ucon)

 return MPCC(mp, G=G, H=H)
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

function cons_nl(mod :: MPCC, x :: Vector)
 return NLPModels.cons(mod.mp, x)
end

function consG(mod :: MPCC, x :: Vector)
 return NLPModels.cons(mod.G, x)
end

function consH(mod :: MPCC, x :: Vector)
 return NLPModels.cons(mod.H, x)
end

#########################################################
#
# Return the vector of the constraints
# lb <= x <= ub, 
# lc <= c(x) <= uc, 
# lccG <= G(x), lccH <= H(x), 
# G(x).*H(x) <= 0
#
#########################################################
function cons(mod :: MPCC, x :: Vector)

 G=consG(mod, x)
 H=consH(mod, x)

 if mod.meta.ncon > 0
  vnl = cons_nl(mod, x)
 else
  vnl = Float64[]
 end

 return vcat(vnl, G, H, G.*H)
end

function cons_mp(mod :: MPCC, x :: Vector)

 if mod.meta.ncon > 0
  vnl = cons_nl(mod, x)
 else
  vnl = Float64[]
 end

 return vcat(x, vnl)
end

function jac_nl(mod :: MPCC, x :: Vector)
 return NLPModels.jac(mod.mp, x)
end

function jacG(mod :: MPCC, x :: Vector)
 return NLPModels.jac(mod.G, x)
end

function jacH(mod :: MPCC, x :: Vector)
 return NLPModels.jac(mod.H, x)
end

function jac(mod :: MPCC, x :: Vector)
 A, Il,Iu,Ig,Ih,IG,IH = jac_actif(mod, x, 0.0)
 return A
end

"""
Jacobienne des contraintes actives à precmpcc près
"""

function jac_actif(mod :: MPCC, x :: Vector, prec :: Float64)

  n = mod.meta.nvar
  ncc = mod.meta.ncc

  Il = find(z->z<=prec,abs.(x-mod.meta.lvar))
  Iu = find(z->z<=prec,abs.(x-mod.meta.uvar))
  jl = zeros(n);jl[Il]=1.0;Jl=diagm(jl);
  ju = zeros(n);ju[Iu]=1.0;Ju=diagm(ju);
  IG=[];IH=[];Ig=[];Ih=[];

 if mod.meta.ncon+ncc ==0

  A=[]

 else

  if mod.meta.ncon > 0
   c = cons_nl(mod,x)
   J = jac_nl(mod, x)
  else
   c = Float64[]
   J = sparse(zeros(0,2))
  end

  Ig=find(z->z<=prec,abs.(c-mod.meta.lcon))
  Ih=find(z->z<=prec,abs.(c-mod.meta.ucon))

  Jg, Jh = zeros(mod.meta.ncon,n), zeros(mod.meta.ncon,n)

  Jg[Ig,1:n] = J[Ig,1:n]
  Jh[Ih,1:n] = J[Ih,1:n]

  if ncc>0

   IG=find(z->z<=prec,abs.(consG(mod,x)-mod.meta.lccG))
   IH=find(z->z<=prec,abs.(consH(mod,x)-mod.meta.lccH))

   JG, JH = zeros(ncc, n), zeros(ncc, n)
   JG[IG,1:n] = jacG(mod,x)[IG,1:n]
   JH[IH,1:n] = jacH(mod,x)[IH,1:n]

   A=[Jl;Ju;-Jg;Jh;-JG;-JH]'
  else
   A=[Jl;Ju;-Jg;Jh]'
  end

 end

 return A, Il,Iu,Ig,Ih,IG,IH
end

function jtprodnl(mod :: MPCC, x :: Vector, v :: Vector)
 return NLPModels.jtprod(mod.mp,x,v)
end

function jtprodG(mod :: MPCC, x :: Vector, v :: Vector)
 return NLPModels.jtprod(mod.G,x,v)
end

function jtprodH(mod :: MPCC, x :: Vector, v :: Vector)
 return NLPModels.jtprod(mod.H,x,v)
end

#hessienne du Lagrangien
function hessnl(mod :: MPCC, x :: Vector ; obj_weight = 1.0, y = zeros)
 return NLPModels.hess(mod.mp,x; obj_weight = obj_weight, y = y)
end

function hessG(mod :: MPCC, x :: Vector ; obj_weight = 1.0, y = zeros)
 return NLPModels.hess(mod.G,x; obj_weight = obj_weight, y = y)
end

function hessH(mod :: MPCC, x :: Vector ; obj_weight = 1.0, y = zeros)
 return NLPModels.hess(mod.H,x; obj_weight = obj_weight, y = y)
end

function hess(mod :: MPCC, x :: Vector ; obj_weight = 1.0, y = zeros)
 if y != zeros NotImplemented() end
 return NLPModels.hess(mod.mp, x)
end

#########################################################
#
# Return the violation of the constraints
# lb <= x <= ub, 
# lc <= c(x) <= uc, 
# lccG <= G(x), lccH <= H(x), 
# G(x).*H(x) <= 0
#
#########################################################
function viol_cons(mod :: MPCC, x :: Vector)

 n = mod.meta.n
 x = length(x) == n ? x : x[1:n]

 feas_x = vcat(max.(mod.meta.lvar-x, 0), max.(x-mod.meta.uvar, 0))

 if mod.meta.ncon !=0

  c = cons_nl(mod, x)
  feas_c = vcat(max.(mod.meta.lcon-c, 0), max.(c-mod.meta.ucon, 0))

 else

  feas_c = Float64[]

 end

 if ncc != 0

  G=consG(mod, x)
  H=consH(mod, x)

  feas_cp = vcat(max.(mod.meta.lccG-G, 0), max.(mod.meta.lccH-H, 0))
  feas_cc = max.(G.*H, 0)
 else
  feas_cp = Float64[]
  feas_cc = Float64[]
 end

 return vcat(feas_x, feas_c, feas_cp, feas_cc)
end

"""
Donne le vecteur de violation des contraintes dans l'ordre : G(x)-yg ; H(x)-yh ; lvar<=x ; x<=uvar ; lvar<=c(x) ; c(x)<=uvar
"""
function viol(mod :: MPCC,
              x   :: Vector,
              yg  :: Vector,
              yh  :: Vector)

 feas_mp = viol_mp(mod, x)

 if mod.meta.ncc > 0

  G = consG(mod, x)
  H = consH(mod, x)
  slack = vcat(G-yg, H-yh)

 else

  slack = Float64[]

 end

 return vcat(slack, feas_mp)
end

function viol(mod :: MPCC, x :: Vector) #x de taille n+2ncc

 n   = mod.meta.nvar
 ncc = mod.meta.ncc

 if length(x) == n

  return vcat(viol_mp(mod,x),viol_comp(mod,x))

 elseif length(x) == n + 2*ncc

  return viol(mod,x[1:n],x[n+1:n+ncc],x[n+ncc+1:n+2*ncc])

 else

  throw(error("Dimension error"))

 end
end

"""
Donne la norme de la violation de la complémentarité min(G,H)
"""
function viol_comp(mod::MPCC,x::Vector)

 n = mod.meta.nvar
 x = length(x)==n?x:x[1:n]

 G=consG(mod,x)
 H=consH(mod,x)

 return mod.meta.ncc>0 ? min.(G, H) : Float64[]
end

function viol_nl(mod   :: MPCC,
                 x     :: Vector)

 n=mod.meta.nvar

 if mod.mp.meta.ncon !=0

  c = cons_nl(mod,x)
  feas_c = vcat(max.(mod.meta.lcon-c,0),max.(c-mod.meta.ucon,0))

 else

  feas_c = Float64[]

 end

 return feas_c
end

function viol_mp(mod   :: MPCC,
                 x     :: Vector)

 n = mod.meta.nvar

 x = length(x)==n ? x : x[1:n]

 feas_x = vcat(max.(mod.meta.lvar-x,0),max.(x-mod.meta.uvar,0))

 feas_c = viol_nl(mod, x)

 return vcat(feas_x, feas_c)
end

"""
Donne la violation de la réalisabilité dual
"""
function dual_feasibility(mod::MPCC,x::Vector,l::Vector,A::Any) #type général pour matrice ? AbstractSparseMatrix

 b=grad(mod,x)

 return A*l+b
end

#end du module
end
