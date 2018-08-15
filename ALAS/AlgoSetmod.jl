module AlgoSetmod

importall Penalty
importall DDirection
importall LineSearch
importall CRhoUpdate

import ActifMPCCmod.actif_solve, ActifMPCCmod.working_min_proj

"""
Commentaires :
Liste les choix algorithmiques utilisés dans la résolution du MPCC :
* choix de la fonction de pénalité
* choix de la direction de descente
* choix de la recherche linéaire

* c(x)*rho tronqué
"""

type AlgoSet

 uncmin      :: Function
 penalty     :: Function
 direction   :: Function
 linesearch  :: Function

 crho_update :: Function #ça sert à quoi déjà ?
end

#Constructeur par défaut
function AlgoSet(;uncmin      :: Function = actif_solve, #working_min_proj
                  penalty     :: Function = Penalty.quadratic, #lagrangian, quadratic
                  direction   :: Function = DDirection.NwtdirectionSpectral, #NwtdirectionSpectral, CGHZ, invBFGS
                  linesearch  :: Function = LineSearch.armijo_wolfe, #armijo, armijo_wolfe, armijo_wolfe_hz
                  crho_update :: Function = CRhoUpdate.MinProd)

 return AlgoSet(uncmin, penalty, direction, linesearch, crho_update)
end

#end of module
end
