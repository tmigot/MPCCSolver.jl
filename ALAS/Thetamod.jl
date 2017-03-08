"""
Package de fonctions qui définissent les fonctions Theta utilisées ici

liste des fonctions :
theta(x::Vector,r::Float64)

invtheta(x::Vector,r::Float64)

dtheta(x::Vector,r::Float64)

AlphaThetaMax(x::Float64,dx::Float64,y::Float64,dy::Float64,r::Float64,s::Float64,t::Float64)
"""

module Thetamod

# TO DO List:
#Minor :
# - fonction AlphaThetaMax illisible à redécouper (+ copie de code)
# - changer le nom du fichier

"""
La fonction theta
theta_r(x)
In : x :: vecteur ; r :: float
"""
function theta(x::Any,r::Float64)
 y=zeros(length(x))
 for i=1:length(x)
  y[i]= x[i]>=0 ? x[i]/(x[i]+r) : x[i]/r-x[i]^2/(r^2)
 end
 return y
end
theta(x::Float64,r::Float64)= x>=0 ? x/(x+r) : x/r-x^2/(r^2)

"""
La fonction inverse de theta
invtheta_r(x)
In : x :: vecteur ; r :: float
"""
function invtheta(x::Any,r::Float64)
 y=zeros(length(x))
 for i=1:length(x)
  y[i]= x[i]>=0 ? -r*x[i]/(x[i]-1) : 1/2*(r-r*sqrt(1-4*x[i]))
 end
 return y
end
invtheta(x::Float64,r::Float64)= x>=0 ? -r*x/(x-1) : 1/2*(r-r*sqrt(1-4*x))

"""
La fonction dérivée theta
dtheta_r(x)
In : x :: vecteur ; r :: float
"""
function dtheta(x::Any,r::Float64)
 y=zeros(length(x))
 for i=1:size(x,1)
  y[i]= x[i]>=0 ? r/(x[i]+r)^2 : 1/r-2*x[i]/(r^2)
 end
 return y
end
dtheta(x::Float64,r::Float64)= x>=0 ? r/(x+r)^2 : 1/r-2*x/(r^2)

"""
Essaye de donner la plus petite racine réelle positive du polynome ax^2+bx+c
"""
function PolyDegre2Positif(a::Float64,b::Float64,c::Float64)
prec=1e-15
 if a!=0
 disc=b^2-4*a*c
  if disc>=0
   sol=min((-b+sqrt(disc))./(2*a),(-b-sqrt(disc))./(2*a))
   if sol<prec #normal si on touche déjà la frontière
    sol=max((-b+sqrt(disc))./(2*a),(-b-sqrt(disc))./(2*a))
   end
  else
   sol=Inf #pas de racines réelles
  end
 else #a==0
  sol=-c./b
 end

 return sol
end
"""
AlphaThetaMax : résoud l'équation en alpha où psi(x,r,s,t)=s+t*theta(x,r) (n=1)
In : x float (position en x)
     dx float (déplacement en x)
     y float (position en y)
     dy float (déplacement en y)
     r float (paramètre)
     s float (paramètre)
     t float (paramètre)
"""
function AlphaThetaMax(x::Float64,dx::Float64,y::Float64,dy::Float64,r::Float64,s::Float64,t::Float64)
 #donner la formule analytique pour Theta1 :
if dx==0 && dy==0
 return [Inf,Inf]
end

 # pour F1 : (Wolfram alpha)
if t!=0

a=dx.*dy
b=dy.*(x-s+t)+dx.*(y-s+r)
c=x.*y-x*s+x*r-y*s+s^2-s*r+t*y-t*s
alphaf1p=PolyDegre2Positif(a,b,c)

 if alphaf1p>=0 && (x+alphaf1p*dx>=s && y+alphaf1p*dy>=s)
  alphaf1=alphaf1p
 elseif alphaf1p>=0 && (x+alphaf1p*dx<=s && y+alphaf1p*dy<=s)
  #contrainte candidate mais trop bas
  a=-dy^2
  b=r^2*dx+r*t*dy-2*y*dy+2*s*dy
  c=r^2*x-r^2*s+r*t*y-r*t*s-y^2-s^2+2*s*y
  alphaf1n=PolyDegre2Positif(a,b,c)

  #alphaf1=alphaf1n devrait être actif si on tient compte de la contrainte "intérieur"
  alphaf1=Inf
 else
  #on ne tape pas cette contrainte
  alphaf1=Inf
 end
else #cas où t=0
 if dx!=0 && ((s-x)./dx>0 || (s==x && dx>0))
  alphaf1=(s-x)./dx
 else
  alphaf1=Inf
 end
end #fin de condition t!=0

# pour F2 : (Wolfram alpha)
if t!=0

a=dx.*dy
b=dx.*(y-s+t)+dy.*(x-s+r)
c=x.*y-y*s+y*r-x*s+s^2-s*r+t*x-t*s
alphaf2p=PolyDegre2Positif(a,b,c)

 if alphaf2p>=0 && (x+alphaf2p*dx>=s && y+alphaf2p*dy>=s)
  #c'est un candidat
  alphaf2=alphaf2p
 elseif alphaf2p>=0 && (x+alphaf2p*dx<=s && y+alphaf2p*dy<=s)
  #contrainte candidate mais trop bas
  a=-dx^2
  b=r^2*dy+r*t*dx-2*x*dx+2*s*dx
  c=r^2*y-r^2*s+r*t*x-r*t*s-x^2-s^2+2*s*x
  alphaf2n=PolyDegre2Positif(a,b,c)

  #alphaf2=alphaf2n devrait être actif si on tient compte de la contrainte "intérieur"
  alphaf2=Inf
 else
  #on ne tape pas cette contrainte
  alphaf2=Inf
 end
else #cas où t=0
 if dy!=0 && ((s-y)./dy>0 || (s==y && dy>0))
  alphaf2=(s-y)./dy
 else
  alphaf2=Inf
 end
end

 #le premier fixe yg et le deuxième yh
 return [alphaf1,alphaf2]
end

#fin du module
end
