module AlgoSetmod

using Penalty
using DDirection
using LineSearch

"""
Liste les choix algorithmiques utilisés dans la résolution du MPCC :
* choix de la fonction de pénalité
* choix de la direction de descente
* choix de la recherche linéaire
"""

type AlgoSet
 penalty::Function
 direction::Function
 linesearch::Function
end

#Constructeur par défaut
function AlgoSet()
 return AlgoSet(Penalty.Quadratic,DDirection.NwtdirectionSpectral,LineSearch.Armijo)
end

#end of module
end
