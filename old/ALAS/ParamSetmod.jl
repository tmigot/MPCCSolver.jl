module ParamSetmod

"""
Liste les paramètres utilisés dans la résolution du MPCC
"""

type ParamSet
 #paramètres pour la résolution du MPCC
 precmpcc::Float64

 #paramètres algorithmiques pour la relaxation
 # le t_bar
 tb::Function #dépend des trois paramètres (r,s,t)
 # l'oracle de choix de epsilon
 prec_oracle::Function #dépend des trois paramètres (r,s,t)
 # la mise à jour des paramètres
 rho_restart::Function #
 paramin::Float64 #valeur minimal pour les paramères (r,s,t)

 #paramètres algorithmiques pour l'activation de contraintes
 ite_max_alas::Int64 #entier >=0
 ite_max_viol::Int64 #entier >=0
 rho_init::Vector #nombre >= 0 ou un vecteur ?
 rho_update::Float64 #nombre >=1
 rho_max::Float64
 goal_viol::Float64 #nombre entre (0,1)

 #paramètres pour la minimisation sans contrainte
 ite_max_armijo::Int64 #nombre maximum d'itération pour la recherche linéaire >= 0
 tau_armijo::Float64 #paramètre pour critère d'Armijo doit être entre (0,0.5)
 armijo_update::Float64 #step=step_0*(1/2)^m
 ite_max_wolfe::Int64
 tau_wolfe::Float64 #entre (tau_armijo,1)
 wolfe_update::Float64

 #paramètres de sortie:
 verbose::Int64
end

#Constructeur par défaut
#nbc est le nombre de contraintes du problème
function ParamSet(nbc::Int64)
 precmpcc=1e-3

 tb=(r,s,t)->-r
 prec_oracle=(r,s,t,prec)->max(r,s,t,prec)
 rho_restart(r,s,t,prec,rho)=rho
 paramin=sqrt(eps(Float64))
 #paramin=1.0

 ite_max_alas=100
 ite_max_viol=10
 rho_init=1*ones(nbc)
 rho_update=1.5 #2.0 bien pour Newton, trop grand sinon.
 rho_max=1/precmpcc #le grand max est 1/eps(Float64)
 goal_viol=0.5

 ite_max_armijo=300
 tau_armijo=0.1 #0.4
 armijo_update=0.9 #0.9
 ite_max_wolfe=10
 tau_wolfe=0.6
 wolfe_update=5.0

 verbose=0 #0 quiet, 1 relaxation, 2 relaxation+activation, 3 relaxation+activation+linesearch

 return ParamSet(precmpcc,tb,prec_oracle,rho_restart,paramin,ite_max_alas,ite_max_viol,rho_init,rho_update,rho_max,goal_viol,ite_max_armijo,tau_armijo,armijo_update,ite_max_wolfe,tau_wolfe,wolfe_update,verbose)
end

#end of module
end
