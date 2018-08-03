module AlgoSetmod

importall Penalty
importall DDirection
importall LineSearch
importall CRhoUpdate

"""
Commentaires :
Liste les choix algorithmiques utilisés dans la résolution du MPCC :
* choix de la fonction de pénalité
* choix de la direction de descente
* choix de la recherche linéaire

* c(x)*rho tronqué
"""

type AlgoSet

 penalty     :: Function
 direction   :: Function
 linesearch  :: Function

 crho_update :: Function #ça sert à quoi déjà ?
end

#Constructeur par défaut
function AlgoSet(;penalty     :: Function = Penalty.lagrangian,
                  direction   :: Function = DDirection.NwtdirectionSpectral,
                  linesearch  :: Function = LineSearch.armijo_wolfe,
                  crho_update :: Function = CRhoUpdate.MinProd)

 return AlgoSet(penalty, direction, linesearch, crho_update)
end

#end of module
end
