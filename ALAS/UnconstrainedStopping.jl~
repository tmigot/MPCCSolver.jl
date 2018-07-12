"""
Package de fonctions pour définir la précision
sur la réalisabilité dual pendant la pénalisation
'sans contraintes'.


liste des fonctions :
Prec(prec::Float64,rho::Vector)
ScalePrec(prec::Float64,rho::Vector)

"""
module UnconstrainedStopping

function Prec(prec::Float64,rho::Vector)
 return prec
end

function ScalePrec(prec::Float64,rho::Vector)
 return max(max.(1./rho),prec)
end

#end of module
end
