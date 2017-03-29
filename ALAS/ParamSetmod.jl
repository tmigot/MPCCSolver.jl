module ParamSetmod

"""
Liste les paramètres utilisés dans la résolution du MPCC
"""

#ATTENTION : rho doit etre un vecteur ?

type ParamSet

 #paramètres algorithmiques pour la relaxation
 # le t_bar ?
 # la mise à jour des paramètres
 # l'oracle de choix de epsilon

 #paramètres algorithmiques pour l'activation de contraintes
 ite_max_alas::Int64 #entier >=0
 ite_max_viol::Int64 #entier >=0
 rho_init::Vector #nombre >= 0 ou un vecteur ?
 rho_update::Float64 #nombre >=1
 goal_viol::Float64 #nombre entre (0,1)

 #paramètres pour la minimisation sans contrainte
 ite_max_armijo::Int64 #nombre maximum d'itération pour la recherche linéaire >= 0
 tau_armijo::Float64 #paramètre pour critère d'Armijo doit être entre (0,0.5)
 armijo_update::Float64 #step=step_0*(1/2)^m
 tau_wolfe::Float64 #entre (tau_armijo,1)
 wolfe_update::Float64
end

#Constructeur par défaut
function ParamSet(nbc::Int64)

 ite_max_alas=10000
 ite_max_viol=20
 rho_init=ones(nbc)
 rho_update=2.0
 goal_viol=0.5

 ite_max_armijo=4000
 tau_armijo=0.4
 armijo_update=0.9
 tau_wolfe=0.9
 wolfe_update=2.0

 return ParamSet(ite_max_alas,ite_max_viol,rho_init,rho_update,goal_viol,ite_max_armijo,tau_armijo,armijo_update,tau_wolfe,wolfe_update)
end

#end of module
end
