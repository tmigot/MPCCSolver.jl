"""
Evalue la fonction objectif d'un MPCC actif : x
"""
function obj(ma::ActifMPCC,x::Vector)

 #increment!(ma, :neval_obj)

# if length(x)==ma.n+2*ma.nb_comp
#  return NLPModels.obj(ma.nlp,x)
# else
#  return NLPModels.obj(ma.nlp,evalx(ma,x))
# end
 xf = evalx(ma,x)
 return NLPModels.obj(ma.nlp,xf)
end

"""
Evalue le gradient de la fonction objectif d'un MPCC actif
x est le vecteur réduit
"""
function grad(ma::ActifMPCC,x::Vector)

 #increment!(ma, :neval_grad)

 #on calcul xf le vecteur complet
 xf=evalx(ma,x)
 #construction du vecteur gradient de taille n+2nb_comp
 gradf=NLPModels.grad(ma.nlp,xf)

 return length(x)==ma.n+2*ma.nb_comp?gradf:grad(ma,x,gradf)
end

"""
Gradient projeté !
x : longeur n+2nb_comp
"""

function grad!(ma::ActifMPCC, x::Vector, gx :: Vector)
 #increment!(nlp, :neval_grad)

 if length(x) == ma.n+2*ma.nb_comp
  gradf=NLPModels.grad(ma.nlp,x)
  gx=grad(ma,x,gradf)
 else
  gx=grad(ma,x)
 end

 return gx
end

"""
 
Fonction qui fait le calcul du gradient

"""

function grad(ma::ActifMPCC,x::Vector,gradf::Vector)

 #on calcul xf le vecteur complet
 xf=evalx(ma,x)
 #construction du vecteur gradient de taille n+2nb_comp
 #gradf=NLPModels.grad(ma.nlp,xf)

 nc = length(ma.wnc)

 gradg=Array{Float64}
 # Conditionnelles pour gérer le cas où w1 et w3 est vide
 #if isempty(ma.w1) && isempty(ma.w3)
 if isempty(ma.w4)
  gradg=zeros(length(ma.w13c))
 #elseif !isempty(ma.w13c) #certaines variables sont fixés
 elseif !isempty(ma.w4) #certaines variables sont fixés
  tmp=zeros(ma.nb_comp)
  #tmp[ma.w3]=Relaxation.dpsi(xf[ma.w3+ma.n+ma.nb_comp],ma.r,ma.s,ma.t).*gradf[ma.w3+ma.n]
  tmp[ma.w4]=Relaxation.dpsi(xf[ma.w4+ma.n],ma.r,ma.s,ma.t).*gradf[ma.w4+ma.nb_comp+ma.n]
  gradg=redd(ma,tmp,ma.w13c)
 else #ma.w13c est vide
  gradg=Float64[]
 end

 gradh=Array{Float64,1}
 #if isempty(ma.w2) && isempty(ma.w4)
 if isempty(ma.w3)
  gradh=zeros(length(ma.w24c))
 #elseif !isempty(ma.w24c)
 elseif !isempty(ma.w3)
  tmp=zeros(ma.nb_comp)
  #tmp[ma.w4]=Relaxation.dpsi(xf[ma.w4+ma.n],ma.r,ma.s,ma.t).*gradf[ma.w4+ma.nb_comp+ma.n]
  tmp[ma.w3]=Relaxation.dpsi(xf[ma.w3+ma.n+ma.nb_comp],ma.r,ma.s,ma.t).*gradf[ma.w3+ma.n]
  gradh=redd(ma,tmp,ma.w24c)
 else
  gradh=Float64[]
 end

 return vcat(gradf[ma.wnc],gradf[ma.w13c+ma.n]+gradg,gradf[ma.w24c+ma.nb_comp+ma.n]+gradh)
end

"""
Evalue la matrice hessienne de la fonction objectif d'un MPCC actif
x est le vecteur réduit
"""
function hess(ma::ActifMPCC,x::Vector)

 #on calcul xf le vecteur complet
 xf=evalx(ma,x)

 #construction de la hessienne de taille (n+2nb_comp)^2

 #H=NLPModels.hess(ma.nlp,xf) #renvoi la triangulaire inférieure tril(H,-1)'
 #H=H+tril(H,-1)'

 H=ma.nlp.H(xf)
 H=H+tril(H,-1)'

 if ma.nb_comp>0
  return hess(ma,x,H)
 else
  return H
 end
end

"""
A partir de la hessienne complète (ou une approximation) : calcul la hessienne dans le sous-espace actif
"""
function hess(ma::ActifMPCC,x::Vector,H::Array{Float64,2})

 #on calcul xf le vecteur complet
 xf=evalx(ma,x)

 nred = length(x)

 nc   = length(ma.wnc)
 nnb  = ma.n+ma.nb_comp
 nnbt = ma.n+2*ma.nb_comp

 #construction du vecteur gradient de taille n+2nb_comp
 gradf=NLPModels.grad(ma.nlp,xf)

 #la hessienne des variables du sous-espace (nredxnred)
 Hred=vcat(hcat(H[ma.wnc,ma.wnc],H[ma.wnc,ma.n+ma.w13c],H[ma.wnc,nnb+ma.w24c]),
           hcat(H[ma.n+ma.w13c,ma.wnc],H[ma.n+ma.w13c,ma.n+ma.w13c],H[ma.n+ma.w13c,nnb+ma.w24c]),
             hcat(H[nnb+ma.w24c,ma.wnc],H[nnb+ma.w24c,ma.n+ma.w13c],H[nnb+ma.w24c,ma.n+ma.nb_comp+ma.w24c]))

 if isempty(ma.w4)
  hessg=sparse(zeros(length(ma.w13c),nred))
 else
  hessg=sparse(zeros(length(ma.w13c),nred)) #Tangi18: Bizarre dimension ?

  w4r=zeros(Int64,length(ma.w4))
  for i=1:length(ma.w4)
   w4r[i]=findfirst(x->x==ma.w4[i],ma.w13c)
  end

  hessg=diagm(Relaxation.ddpsi(xf[ma.w4+ma.n],ma.r,ma.s,ma.t).*gradf[ma.w4+ma.nb_comp+ma.n])
  hessg+=diagm(Relaxation.dpsi(xf[ma.w4+ma.n],ma.r,ma.s,ma.t))*H[ma.w4+ma.nb_comp+ma.n,ma.w4+ma.nb_comp+ma.n]

  Hred[nc+w4r,nc+w4r]+=hessg
 end

 if isempty(ma.w3)
  hessh=sparse(zeros(length(ma.w24c),nred))
 else
  hessh=sparse(zeros(length(ma.w24c),nred))

  w3r=zeros(Int64,length(ma.w3))
  for i=1:length(ma.w3)
   w3r[i]=findfirst(x->x==ma.w3[i],ma.w24c)
  end
  hessh=diagm(Relaxation.ddpsi(xf[ma.w3+ma.nb_comp+ma.n],ma.r,ma.s,ma.t).*gradf[ma.w3+ma.n])
  hessh+=diagm(Relaxation.dpsi(xf[ma.w3+ma.nb_comp+ma.n],ma.r,ma.s,ma.t))*H[ma.w3+ma.n,ma.w3+ma.n]

  Hred[nc+length(ma.w13c)+w3r,nc+length(ma.w13c)+w3r]+=hessh
 end

 return Hred
end

"""

Vérifie la violation des contraintes actives

x in n+2nb_comp

"""
function cons(ma::ActifMPCC,x::Vector)

 #increment!(ma, :neval_cons)
 xf = evalx(ma,x)

 sg = xf[ma.n+1:ma.n+ma.nb_comp]
 sh = xf[ma.n+ma.nb_comp+1:ma.n+2*ma.nb_comp]

 vlx = xf[ma.wn1]-ma.nlp.meta.lvar[ma.wn1]
 vux = xf[ma.wn2]-ma.nlp.meta.lvar[ma.wn2]

 vlg = sg[ma.w1]-ma.nlp.meta.lvar[ma.w1+ma.n]
 vlh = sh[ma.w2]-ma.nlp.meta.lvar[ma.w2+ma.n]
 vug = Relaxation.psi(sh[ma.w3],ma.r,ma.s,ma.t)-sg[ma.w3]
 vuh = Relaxation.psi(sg[ma.w4],ma.r,ma.s,ma.t)-sh[ma.w4]

 return minimum([vlx;vux;vlg;vlh;vug;vuh;0.0])==0.0
end

function cons!(ma::ActifMPCC, x :: Vector, cx :: Bool)

 #increment!(ma, :neval_cons)

 cx = cons(ma,x)

 return cx
end
