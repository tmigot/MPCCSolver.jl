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

 # dans ALAS le critère d'arrêt sur le gradient du Lagrangien est donné par :
 # dual_feasible/scaling_dual<=prec
 scaling_dual::Function #dépend des multiplicateurs, la précision(r,s,t),
			#la précision demandé et le gradient du lagrangien
 unconstrained_stopping::Function
end

#Constructeur par défaut
function AlgoSet()

  #scaling_dual=(usg,ush,uxl,uxu,ucl,ucu,lg,lh,lphi,precrst,prec,rho,dualfeas)->max(norm([usg;ush;uxl;uxu;ucl;ucu;lg;lh;lphi]),1)
  scaling_dual=(usg,ush,uxl,uxu,ucl,ucu,lg,lh,lphi,precrst,prec,rho,dualfeas)->1
  #précision du problème pénalisé "sans contraintes"
  #unconstrained_stopping=(prec,rho)->max(maximum(1./rho),prec)
  unconstrained_stopping=(prec,rho)->prec

 return AlgoSet(Penalty.Quadratic,DDirection.NwtdirectionSpectral,LineSearch.Armijo,scaling_dual,unconstrained_stopping)
end

function AlgoSet(penalty::Function,direction::Function, linesearch::Function)

  scaling_dual=(usg,ush,uxl,uxu,ucl,ucu,lg,lh,lphi,precrst,prec,rho,dualfeas)->max(norm([usg;ush;uxl;uxu;ucl;ucu;lg;lh;lphi]),1)
  unconstrained_stopping=(prec,rho)->max(max.(1./rho),prec)

 return AlgoSet(penalty,direction, linesearch,scaling_dual,unconstrained_stopping)
end

#end of module
end
