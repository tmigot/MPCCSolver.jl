"""
Package de fonctions qui définissent les fonctions Theta utilisées ici

liste des fonctions :
theta(x::Vector,r::Float64)

invtheta(x::Vector,r::Float64)

dtheta(x::Vector,r::Float64)

AlphaThetaMax(x::Float64,dx::Float64,y::Float64,dy::Float64,r::Float64,s::Float64,t::Float64)
AlphaThetaMaxOneLeaf(x::Float64,dx::Float64,y::Float64,dy::Float64,r::Float64,s::Float64,t::Float64)
"""

module Thetafunc

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
La fonction dérivée seconde theta
ddtheta_r(x)
In : x :: vecteur ; r :: float
"""
function ddtheta(x::Any,r::Float64)
 y=zeros(length(x))
 for i=1:size(x,1)
  y[i]= x[i]>=0 ? -2*r/(x[i]+r)^3 : -2/(r^2)
 end
 return y
end
ddtheta(x::Float64,r::Float64)= x>=0 ? -2*r/(x+r)^3 : -2/(r^2)

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
if dx==0.0 && dy==0.0
 return [Inf,Inf]
end

alphaf1=AlphaThetaMaxOneLeaf(x,dx,y,dy,r,s,t)

alphaf2=AlphaThetaMaxOneLeaf(y,dy,x,dx,r,s,t)

 #le premier fixe yg et le deuxième yh
 return [alphaf1,alphaf2]
end
"""
Résoud l'équation en a:
(x+a*dx-s)*(y+a*dy-s+r)-t*(y+a*dy-s)=0 -> alphaf1p
et si besoin résoud l'équation en a:
r^2[x+a*dx-s-t*((y+a*dy-s)/r-(y+a*dy-s)^2/r^2)]=0 -> alphaf1n
"""
function AlphaThetaMaxOneLeaf(x::Float64,dx::Float64,y::Float64,dy::Float64,r::Float64,s::Float64,t::Float64)

 if t!=0

 #a=dx.*dy erreur de calcul
 #b=dy.*(x-s+t)+dx.*(y-s+r)
 #c=x.*y-x*s+x*r-y*s+s^2-s*r+t*y-t*s
 a=dx.*dy
 b=dy.*(x-s-t)+dx.*(y-s+r)
 c=x.*y-x*s+x*r-y*s+s^2-s*r-t*y+t*s
 alphaf1p=PolyDegre2Positif(a,b,c)

  if alphaf1p>=0 && (x+alphaf1p*dx>=s && y+alphaf1p*dy>=s)
   alphaf1=alphaf1p
  elseif alphaf1p>=0 && (x+alphaf1p*dx<=s && y+alphaf1p*dy<=s)
   #contrainte candidate mais trop bas
   #a=-dy^2 erreur de calcul
   #b=r^2*dx+r*t*dy-2*y*dy+2*s*dy
   #c=r^2*x-r^2*s+r*t*y-r*t*s-y^2-s^2+2*s*y
   a=dy^2
   b=r^2*dx-r*t*dy+t*2*y*dy-t*2*s*dy
   c=r^2*x-r^2*s-r*t*y+r*t*s+t*y^2+t*s^2-t*2*s*y
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

 return alphaf1
end

#fin du module
end
