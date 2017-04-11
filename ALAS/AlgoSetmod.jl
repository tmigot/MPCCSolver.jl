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

 #dans ALAS le critère d'arrêt sur le gradient du Lagrangien est donné par :
 # dual_feasible/scaling_dual<=prec
 scaling_dual::Function #dépend des multiplicateurs, la précision(r,s,t), la précision demandé et le gradient du lagrangien
end

#Constructeur par défaut
function AlgoSet()

  scaling_dual=(usg,ush,uxl,uxu,ucl,ucu,lg,lh,lphi,precrst,prec,rho,dualfeas)->max(norm([usg;ush;uxl;uxu;ucl;ucu;lg;lh;lphi]),1)

 return AlgoSet(Penalty.Quadratic,DDirection.CGHZ,LineSearch.ArmijoWolfe,scaling_dual)
end

#end of module
end
