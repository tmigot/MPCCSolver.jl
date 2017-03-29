module AlgoSetmod

using DDirection
using LineSearch

"""
Liste les choix algorithmiques utilisés dans la résolution du MPCC :
* choix de la fonction de pénalité
* choix de la direction de descente
* choix de la recherche linéaire
* choix de la mise à jour de rho
"""

type AlgoSet
 penalty::Function
 direction::Function
 linesearch::Function
 rhoinit::Function
end

#Constructeur par défaut
function AlgoSet()
 return AlgoSet(DDirection.CGHZ,DDirection.CGHZ,LineSearch.Armijo,DDirection.CGHZ)
end

#end of module
end
