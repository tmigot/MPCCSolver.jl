"""
Package de fonctions qui définissent la relaxation

liste des fonctions :
psi(x::Any,r::Float64,s::Float64,t::Float64)
invpsi(x::Any,r::Float64,s::Float64,t::Float64)
dpsi(x::Any,r::Float64,s::Float64,t::Float64)

phi(yg::Any,yh::Any,r::Float64,s::Float64,t::Float64)
phi(x::Any,nb_comp::Int64,r::Float64,s::Float64,t::Float64)

dphi(yg::Any,yh::Any,r::Float64,s::Float64,t::Float64)
dphi(x::Any,r::Float64,s::Float64,t::Float64)

AlphaThetaMax(x::Float64,dx::Float64,y::Float64,dy::Float64,r::Float64,s::Float64,t::Float64)
"""
module Relaxation

using Thetafunc

"""
psi(x) : colle au notation de l'article pour la méthode des papillons
"""
function psi(x::Any,r::Float64,s::Float64,t::Float64)
 return s+t*Thetafunc.theta(x-s,r)
end

function invpsi(x::Any,r::Float64,s::Float64,t::Float64)
 if t==0
#  println("Error invpsi: inverse not defined")
 end
 return t!=0?s+Thetafunc.invtheta((x-s)/t,r):-Inf
end

function dpsi(x::Any,r::Float64,s::Float64,t::Float64)
 return t*Thetafunc.dtheta(x-s,r)
end

function ddpsi(x::Any,r::Float64,s::Float64,t::Float64)
 return t*Thetafunc.ddtheta(x-s,r)
end

"""
phi(x,r,s,t) : évalue la fonction de relaxation de la contrainte de complémentarité en (yg,yh) in (nb_comp,nb_comp)
"""
function phi(yg::Any,yh::Any,r::Float64,s::Float64,t::Float64)
 return (yg-psi(yh,r,s,t)).*(yh-psi(yg,r,s,t))
end
"""
phi(x,r,s,t) : évalue la fonction de relaxation de la contrainte de complémentarité en x in n+2nb_comp
"""
function phi(x::Any,nb_comp::Int64,r::Float64,s::Float64,t::Float64)
 n=length(x)-2*nb_comp
 yg=x[n+1:n+mod.nb_comp]
 yh=x[n+mod.nb_comp+1:n+2*mod.nb_comp]
 return phi(yg,yh,r,s,t)
end

"""
dphi(x,r,s,t) : évalue la fonction de relaxation de la contrainte de complémentarité en (yg,yh) in (nb_comp,nb_comp)
"""
function dphi(yg::Any,yh::Any,r::Float64,s::Float64,t::Float64)
 dg=(yh-psi(yg,r,s,t))-dpsi(yg,r,s,t).*(yg-psi(yh,r,s,t))
 dh=(yg-psi(yh,r,s,t))-dpsi(yh,r,s,t).*(yh-psi(yg,r,s,t))
 return [dg;dh]
end

"""
dphi(yg,yh,r,s,t) : évalue la fonction de relaxation de la contrainte de complémentarité en x in n+2nb_comp
"""
function dphi(x::Any,r::Float64,s::Float64,t::Float64)
 n=length(x)-2*nb_comp
 yg=x[n+1:n+mod.nb_comp]
 yh=x[n+mod.nb_comp+1:n+2*mod.nb_comp]
 return [zeros(n);dphi(yg,yh,r,s,t)]
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
 return Thetafunc.AlphaThetaMax(x,dx,y,dy,r,s,t)
end

#end of module
end
