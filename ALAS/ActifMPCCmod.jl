module ActifMPCCmod

import Relaxation
import ParamSetmod
importall NLPModels
#NLPModels.AbstractNLPModel,  NLPModelMeta, Counters

"""
Type ActifMPCC : problème MPCC pénalisé avec slack et ensemble des contraintes actives

liste des constructeurs :
ActifMPCC(nlp::NLPModels.AbstractNLPModel,r::Float64,s::Float64,t::Float64,nb_comp::Int64,paramset::ParamSetmod.ParamSet,direction::Function,linesearch::Function)
ActifMPCC(nlp::NLPModels.AbstractNLPModel,r::Float64,s::Float64,t::Float64,w::Array{Bool,2},paramset::ParamSetmod.ParamSet,direction::Function,linesearch::Function)

liste des méthodes :
updatew(ma::ActifMPCC)
setw(ma::ActifMPCC, w::Array{Bool,2})
setbeta(ma::ActifMPCC,b::Float64)
sethess(ma::ActifMPCC,Hess::Array{Float64,2})

liste des accesseurs :
evalx(ma::ActifMPCC,x::Vector)
evald(ma::ActifMPCC,d::Vector)
redd(ma::ActifMPCC,d::Vector,w::Array{Int64,1})
obj(ma::ActifMPCC,x::Vector)
ExtddDirection(ma::ActifMPCCmod.ActifMPCC,dr::Vector,xp::Vector,step::Float64)
grad(ma::ActifMPCC,x::Vector)
grad(ma::ActifMPCC,x::Vector,gradf::Vector)
hess(ma::ActifMPCC,x::Vector)
hess(ma::ActifMPCC,x::Vector,H::Array{Float64,2})

liste des fonctions :
LSQComputationMultiplier(ma::ActifMPCC,gradpen::Vector,xj::Vector)
RelaxationRule(ma::ActifMPCCmod.ActifMPCC,xj::Vector,lg::Vector,lh::Vector,lphi::Vector,wmax::Array{Bool,2})
PasMaxComp(ma::ActifMPCC,x::Vector,d::Vector)
AlphaChoix(alpha::Float64,alpha1::Float64,alpha2::Float64)
AlphaChoix(alpha::Float64,alpha1::Float64,alpha2::Float64,alpha3::Float64,alpha4::Float64)
PasMaxBound(ma::ActifMPCC,x::Vector,d::Vector) #TO DO
PasMax(ma::ActifMPCC,x::Vector,d::Vector)
"""

#TO DO List :
#Major :
#- intégrer les contraintes de bornes sur x dans PasMax
#Minor :
#- copie de code dans PasMax

type ActifMPCC <: AbstractNLPModel

 meta :: NLPModelMeta
 counters :: Counters #ATTENTION : increment! ne marche pas?
 x0 :: Vector

 nlp :: AbstractNLPModel # en fait on demande juste une fonction objectif, point initial, contraintes de bornes
 r::Float64
 s::Float64
 t::Float64

 w::Array{Bool,2} # n +2nb_comp x 2 matrix

 n::Int64 #dans le fond est optionnel si on a nb_comp
 nb_comp::Int64

 #set of indices:
 wnc :: Array{Int64,1} #indices of free variables in 1,...,n
 wn1::Array{Int64,1} # indices with active lower bounds in 1,...,n
 wn2::Array{Int64,1} # indices with active upper bounds in 1,...,n

 #ensembles d'indices complementarity:
 w1::Array{Int64,1} # ensemble des indices (entre 0 et nb_comp) où la contrainte yG>=-r est active
 w2::Array{Int64,1} # ensemble des indices (entre 0 et nb_comp) où la contrainte yH>=-r est active
 w3::Array{Int64,1} # ensemble des indices (entre 0 et nb_comp) où la contrainte yG<=s+t*theta(yH,r) est active
 w4::Array{Int64,1} # ensemble des indices (entre 0 et nb_comp) où la contrainte yH<=s+t*theta(yG,r) est active
 wcomp::Array{Int64,1} #ensemble des indices (entre 0 et nb_comp) où la contrainte Phi<=0 est active
 w13c::Array{Int64,1} #ensemble des indices où les variables yG sont libres
 w24c::Array{Int64,1} #ensemble des indices où les variables yH sont libres
 wc::Array{Int64,1} #ensemble des indices des contraintes où yG et yH sont libres
 wcc::Array{Int64,1} #ensemble des indices des contraintes où yG et yH sont fixés


 wnew::Array{Bool,2} #dernières contraintes ajoutés

 #paramètres pour le calcul de la direction de descente
 crho::Float64 #constant such that : ||c(x)||_2 \approx crho*rho
 beta::Float64 #paramètre pour gradient conjugué
 Hess::Array{Float64,2} #inverse matrice hessienne approximée
 #Hd::Vector #produit inverse matrice hessienne et gradient (au lieu de la hessienne entière)

 paramset::ParamSetmod.ParamSet
 direction::Function #fonction qui calcul la direction de descente
 linesearch::Function #fonction qui calcul la recherche linéaire

end

"""
Constructeur recommandé pour ActifMPCC
"""
#Tangi18: Pourquoi c'est coupé en 2 ?
function ActifMPCC(nlp::NLPModels.AbstractNLPModel,
                   r::Float64,s::Float64,t::Float64,
                   nb_comp::Int64,
                   paramset::ParamSetmod.ParamSet,
                   direction::Function,
                   linesearch::Function)

 nn  = length(nlp.meta.x0) #n + 2nb_comp
 n   = length(nlp.meta.x0)-2*nb_comp
 xk  = nlp.meta.x0[1:n]
 ygk = nlp.meta.x0[n+1:n+nb_comp]
 yhk = nlp.meta.x0[n+nb_comp+1:n+2*nb_comp]

 w   = zeros(Bool,nn,2)

 #active bounds
 w[1:nn,1]= nlp.meta.x0 .== nlp.meta.lvar
 w[1:n,2]= nlp.meta.x0[1:n] .== nlp.meta.uvar[1:n]

  # puis la boucle: est-ce qu'il y a Relaxation.psi vectoriel ?
  #A simplifier
  for l=1:nb_comp
   if ygk[l]==Relaxation.psi(yhk[l],r,s,t)
    w[n+l+nb_comp,1]=true;
   end
   if yhk[l]==Relaxation.psi(ygk[l],r,s,t)
    w[n+l+nb_comp,2]=true;
   end
  end

 return ActifMPCC(nlp,r,s,t,nb_comp,n,w,paramset,direction,linesearch)
end

function ActifMPCC(nlp::NLPModels.AbstractNLPModel,
                   r::Float64,s::Float64,t::Float64,
                   nb_comp::Int64, 
                   n :: Int64,
                   w::Array{Bool,2},
                   paramset::ParamSetmod.ParamSet,
                   direction::Function,
                   linesearch::Function)


 wnc = find(.!w[1:n,1] .& .!w[1:n,2])
 wn1=find(w[1:n,1])
 wn2=find(w[1:n,2])

 w1=find(w[n+1:n+nb_comp,1])
 w2=find(w[n+1:n+nb_comp,2])
 w3=find(w[n+nb_comp+1:n+2*nb_comp,1])
 w4=find(w[n+nb_comp+1:n+2*nb_comp,2])

 wcomp=find(w[n+nb_comp+1:n+2*nb_comp,1] .| w[n+nb_comp+1:n+2*nb_comp,2])
 w13c=find(.!w[n+1:n+nb_comp,1] .& .!w[n+nb_comp+1:n+2*nb_comp,1])
 w24c=find(.!w[n+1:n+nb_comp,2] .& .!w[n+nb_comp+1:n+2*nb_comp,2])
 wc=find(.!w[n+1:n+nb_comp,1] .& .!w[n+1:n+nb_comp,2] .& .!w[n+nb_comp+1:n+2*nb_comp,1] .& .!w[n+nb_comp+1:n+2*nb_comp,2])
 wcc=find((w[n+1:n+nb_comp,1] .| w[n+nb_comp+1:n+2*nb_comp,1]) .& (w[n+1:n+nb_comp,2] .| w[n+nb_comp+1:n+2*nb_comp,2]))

 wnew=zeros(Bool,0,0)

 crho=1.0
 beta=0.0
 Hess=eye(n+2*nb_comp)

 
 meta = nlp.meta
 x = nlp.meta.x0
 x0 = [x[1:n];x[n+w13c];x[n+nb_comp+w24c]]

 return ActifMPCC(meta,Counters(),x0,nlp,r,s,t,w,n,nb_comp,
                  wnc,wn1,wn2,w1,w2,w3,w4,
                  wcomp,w13c,w24c,wc,wcc,wnew,crho,beta,Hess,
                  paramset,direction,linesearch)
end

"""
Methodes pour le ActifMPCC
"""
#Mise à jour de w
function setw(ma::ActifMPCC, w::Array{Bool,2})

 ma.wnew=w .& .!ma.w
 ma.w=w

 return updatew(ma)
end

#Mise à jour des composantes liés à w
function updatew(ma::ActifMPCC)

 nb_comp = ma.nb_comp
 n = ma.n

 #le vecteur x d'avant (ne dépend pas de ma.w)
 x = evalx(ma,ma.x0)

 #on actualise avec w
 ma.wnc = find(.!ma.w[1:n,1] .& .!ma.w[1:n,2])
 ma.wn1=find(ma.w[1:n,1])
 ma.wn2=find(ma.w[1:n,2])

 ma.w1=find(ma.w[n+1:n+nb_comp,1])
 ma.w2=find(ma.w[n+1:n+nb_comp,2])
 ma.w3=find(ma.w[n+nb_comp+1:n+2*nb_comp,1])
 ma.w4=find(ma.w[n+nb_comp+1:n+2*nb_comp,2])

 ma.wcomp=find(ma.w[n+nb_comp+1:n+2*nb_comp,1] .| ma.w[n+nb_comp+1:n+2*nb_comp,2])

 ma.w13c=find(.!ma.w[n+1:n+nb_comp,1] .& .!ma.w[n+nb_comp+1:n+2*nb_comp,1])
 ma.w24c=find(.!ma.w[n+1:n+nb_comp,2] .& .!ma.w[n+nb_comp+1:n+2*nb_comp,2])

 ma.wc=find(.!ma.w[n+1:n+nb_comp,1] .& .!ma.w[n+1:n+nb_comp,2] .& .!ma.w[n+nb_comp+1:n+2*nb_comp,1] .& .!ma.w[n+nb_comp+1:n+2*nb_comp,2])

 ma.wcc=find((ma.w[n+1:n+nb_comp,1] .| ma.w[n+nb_comp+1:n+2*nb_comp,1]) .& (ma.w[n+1:n+nb_comp,2] .| ma.w[n+nb_comp+1:n+2*nb_comp,2]))

 ma.x0 = [x[ma.wnc];x[ma.n+ma.w13c];x[ma.n+ma.nb_comp+ma.w24c]]

 return ma
end

function setbeta(ma::ActifMPCC,b::Float64)
 ma.beta=b
 return ma
end

function setcrho(ma::ActifMPCC,crho::Float64)
 ma.crho=crho
 return ma
end

function sethess(ma::ActifMPCC,Hess::Array{Float64,2})
 ma.Hess=Hess
 return ma
end

"""
Renvoie le vecteur x=[x,yg,yh] au complet
"""
function evalx(ma::ActifMPCC,x::Vector)

 if length(x) != ma.n+2*ma.nb_comp

  nc = length(ma.wnc)
  nw13c = nc+length(ma.w13c)

  #construction du vecteur de taille n+2nb_comp que l'on évalue :
  xf=ma.s*ones(ma.n+2*ma.nb_comp)
  xf[ma.wnc]=x[1:nc]
  xf[ma.w13c+ma.n]=x[nc+1:nw13c]
  xf[ma.w24c+ma.n+ma.nb_comp]=x[nw13c+1:nw13c+length(ma.w24c)]

  #on regarde les variables x fixées:
  xf[ma.wn1] = ma.nlp.meta.lvar[ma.wn1]
  xf[ma.wn2] = ma.nlp.meta.uvar[ma.wn2]

  #on regarde les variables yG fixées :
  xf[ma.w1+ma.n]=ma.nlp.meta.lvar[ma.w1+ma.n]
  xf[ma.w3+ma.n]=Relaxation.psi(xf[ma.w3+ma.n+ma.nb_comp],ma.r,ma.s,ma.t)

  #on regarde les variables yH fixées :
  xf[ma.w2+ma.n+ma.nb_comp]=ma.nlp.meta.lvar[ma.w2+ma.n+ma.nb_comp]
  xf[ma.w4+ma.n+ma.nb_comp]=Relaxation.psi(xf[ma.w4+ma.n],ma.r,ma.s,ma.t)

 else

  xf = x

 end

 return xf
end

"""
Renvoie la direction d au complet (avec des 0 aux actifs)
"""
function evald(ma::ActifMPCC,d::Vector)

 nc = length(ma.wnc)
 nw13c = nc+length(ma.w13c)

 df=zeros(ma.n+2*ma.nb_comp)
 df[ma.wnc]=d[1:nc]
 df[ma.w13c+ma.n]=d[nc+1:nw13c]
 df[ma.w24c+ma.nb_comp+ma.n]=d[nw13c+1:nw13c+length(ma.w24c)]

 return df
end

"""
Renvoie la direction d réduite
"""
function redd(ma::ActifMPCC,d::Vector)

 nc = length(ma.wnc)
 nw13c = nc+length(ma.w13c)

 df=zeros(nc+length(ma.w13c)+length(ma.w24c))
 df[1:nc]=d[ma.wnc]
 df[nc+1:nw13c]=d[ma.w13c+ma.n]
 df[nw13c+1:nw13c+length(ma.w24c)]=d[ma.w24c+ma.nb_comp+ma.n]

 return df
end

function redd(ma::ActifMPCC,d::Vector,w::Array{Int64,1})

  df=zeros(length(w))
  df[1:length(w)]=d[w]

 return df
end

"""
Calcul la direction étendue à partir de la direction du sous-domaine
complète une version étendue de la direction de descente :
dr : la direction dans le sous-espace actif
xp : le nouvel itéré
step : le pas utilisé pour calculé xp

utilise la formule suivante :
yG+=yG+alpha*dyG --> dyG=(yG+-yG)/alpha
"""
#Tangi18: Est-ce que cette fonction sert à quelque chose ?
function ExtddDirection(ma::ActifMPCCmod.ActifMPCC,
                        dr::Vector,xp::Vector,step::Float64)
@show "ExtddDirection: Est-ce que je sers à quelque chose ?"
 d=evald(ma,dr) #evald rempli les trous par des 0
 x=evalx(ma,xp)

 d[1:ma.n] = dr[1:ma.n]

 psip = Relaxation.psi(x[ma.n+1:ma.n+2*ma.nb_comp],ma.r,ma.s,ma.t)
 psi  = Relaxation.psi(x[ma.n+1:ma.n+2*ma.nb_comp]-step*d[ma.n+1:ma.n+2*ma.nb_comp],
                    ma.r,ma.s,ma.t)
 #d[ma.w1]=0 #yG fixé
 #d[ma.w2]=0 #yH fixé

 d[ma.n+ma.w3]=(psip[ma.w3+ma.nb_comp]-psi[ma.w3+ma.nb_comp])/step #yG fixé
 d[ma.n+ma.nb_comp+ma.w4]=(psip[ma.w4]-psi[ma.w4])/step #yH fixé
 #d[ma.n+ma.w13c]=dr[ma.n+ma.w13c] #yG est libre
 #d[ma.n+ma.nb_comp+ma.w24c]=dr[ma.n+ma.nb_comp+ma.w24c] #yG est libre

 return d
end

############################################################################
#
# Classical NLP functions on ActifMPCC
# obj, grad, grad!, hess, cons, cons!
#
############################################################################

include("actifmpcc_nlp.jl")

############################################################################
#LSQComputationMultiplier(ma::ActifMPCC,x::Vector) :
#ma ActifMPCC
#xj in n+2nb_comp
#gradpen in n+2nb_comp
#
#calcul la valeur des multiplicateurs de Lagrange pour la contrainte de complémentarité en utilisant moindre carré
############################################################################

function LSQComputationMultiplierBool(ma::ActifMPCC,
                                      gradpen::Vector,
                                      xjk::Vector)

   l = LSQComputationMultiplier(ma,gradpen,xjk)

   l_negative = findfirst(x->x<0,l)!=0

 return l,l_negative
end

function LSQComputationMultiplier(ma::ActifMPCC,
                                  gradpen::Vector,
                                  xj::Vector)

 dg = Relaxation.dphi(xj[ma.n+1:ma.n+ma.nb_comp],
                      xj[ma.n+ma.nb_comp+1:ma.n+2*ma.nb_comp],
                      ma.r,ma.s,ma.t)

 gx = dg[1:ma.nb_comp]
 gy = dg[ma.nb_comp+1:2*ma.nb_comp]

 #matrices des contraintes actives : (lx,ux,lg,lh,lphi)'*A=b
 nx1 = length(ma.wn1)
 nx2 = length(ma.wn2)
 nx  = nx1 + nx2

 nw1 = length(ma.w1)
 nw2 = length(ma.w2)
 nwcomp = length(ma.wcomp)

 Dlg = -diagm(ones(nw1))
 Dlh = -diagm(ones(nw2))
 Dlphig = diagm(collect(gx)[ma.wcomp])
 Dlphih = diagm(collect(gy)[ma.wcomp])

 #matrix of size: nc+nw1+nw2+nwcomp x nc+nw1+nw2+2nwcomp
 A=[hcat(-diagm(ones(nx1)),zeros(nx1,nx2+nw1+nw2+2*nwcomp));
    hcat(zeros(nx2,nx1),diagm(ones(nx2)),zeros(nx2,nw1+nw2+2*nwcomp));
    hcat(zeros(nw1,nx),Dlg,zeros(nw1,nw2+2*nwcomp));
    hcat(zeros(nw2,nx),zeros(nw2,nw1),Dlh,zeros(nw2,2*nwcomp));
    hcat(zeros(nwcomp,nx),zeros(nwcomp,nw1+nw2),Dlphig,Dlphih)]

 #vector of size: nc+nw1+nw2+2nwcomp
 b=-[gradpen[ma.wn1];
     gradpen[ma.wn2];
     gradpen[ma.w1+ma.n];
     gradpen[ma.wcomp+ma.n];
     gradpen[ma.w2+ma.n+ma.nb_comp];
     gradpen[ma.wcomp+ma.n+ma.nb_comp]] 

 #compute the multiplier using pseudo-inverse
 l=pinv(A')*b

 lk = zeros(2*ma.n+3*ma.nb_comp)
 lk[ma.wn1] = l[1:nx1]
 lk[ma.n+ma.wn2] = l[nx1+1:nx]
 lk[2*ma.n+ma.w1] = l[nx+1:nx+nw1]
 lk[2*ma.n+ma.nb_comp+ma.w2] = l[nx+nw1+1:nx+nw1+nw2]
 lk[2*ma.n+2*ma.nb_comp+ma.wcomp] = l[nx+nw1+nw2+1:nx+nw1+nw2+nwcomp]

 return lk
end

############################################################################
#
# Define the relaxation rule over the active constraints
#
#function RelaxationRule(ma :: ActifMPCC,
#                        xj :: Vector,
#                        l :: Vector,
#                        wmax :: Array{Bool,2})
#
# Input:
#     xj          : vector of size n+2nb_comp
#     l           : vector of multipliers size (n + 2nb_comp)
#     wmax        : set of freshly added constraints
#
#return: ActifMPCC (with updated active constraints)
#
############################################################################

function RelaxationRule(ma :: ActifMPCC,
                        xj :: Vector,
                        l :: Vector,
                        wmax :: Array{Bool,2})

  copy_wmax = copy(wmax)

  llx  = l[1:ma.n]
  lux  = l[ma.n+1:2*ma.n]
  lg   = l[2*ma.n+1:2*ma.n+ma.nb_comp]
  lh   = l[2*ma.n+ma.nb_comp+1:2*ma.n+2*ma.nb_comp]
  lphi = l[2*ma.n+2*ma.nb_comp+1:2*ma.n+3*ma.nb_comp]

  # Relaxation de l'ensemble d'activation : 
  # désactive toutes les contraintes négatives
  ll = [llx;lg;lphi;lux;lh;lphi] #pas très catholique comme technique
  ma.w[find(x -> x<0,ll)] = zeros(Bool,length(find(x -> x<0,ll)))

  # Règle d'anti-cyclage : 
  # on enlève pas une contrainte qui vient d'être ajouté.
  ma.w[find(x->x==1.0,copy_wmax)] = ones(Bool,length(find(x->x==1.0,copy_wmax)))

 return ActifMPCCmod.updatew(ma)
end

############################################################################
#
# Compute the maximum step to stay feasible in a direction
#
# PasMax(ma::ActifMPCC,x::Vector,d::Vector)
#
# return: alpha, (the step)
#         w_save, (contraintes actives au point x+step*d)
#         w_new  (les nouvelles contraintes actives au point x+step*d)
#
############################################################################

include("actifmpcc_pasmax.jl")

#end of module
end
