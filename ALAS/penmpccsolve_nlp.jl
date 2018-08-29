"""
Evalue la fonction objectif d'un MPCC actif : x
"""
function obj(ma :: PenMPCCSolve, x :: Vector)

 #increment!(ma, :neval_obj)
@show "On vient la???"
 return obj(ma.pen, x)
end

"""
Evalue le gradient de la fonction objectif d'un MPCC actif
x est le vecteur réduit
"""
function grad(ma :: PenMPCCSolve,x :: Vector)

 #increment!(ma, :neval_grad)

 return grad(ma.pen, x)
end
