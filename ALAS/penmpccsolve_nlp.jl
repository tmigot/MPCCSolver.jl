"""
Evalue la fonction objectif d'un MPCC actif : x
"""
function obj(ma :: PenMPCCSolve,x :: Vector)

 #increment!(ma, :neval_obj)

 return NLPModels.obj(ma.pen.nlp, x)
end

"""
Evalue le gradient de la fonction objectif d'un MPCC actif
x est le vecteur réduit
"""
function grad(ma :: PenMPCCSolve,x :: Vector)

 #increment!(ma, :neval_grad)

 return NLPModels.grad(ma.pen.nlp, x)
end
