module ParamSetmod

"""
Liste les paramètres utilisés dans la résolution du MPCC:
ParamSet(nbc::Int64)
ParamSet(nbc::Int64,r0::Float64,sigma_r::Float64,s0::Float64,sigma_s::Float64,t0::Float64,sigma_t::Float64)
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
 initrst::Function #initialize (r,s,t)
 updaterst::Function #update (r,s,t)

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

 (sigma_r,sigma_s,sigma_t)=(0.1,0.1,0.05)
 (r0,s0,t0)=(1.0,0.5,0.5)
 return ParamSet(nbc,r0,sigma_r,s0,sigma_s,t0,sigma_t)
end

function ParamSet(nbc::Int64,r0::Float64,sigma_r::Float64,s0::Float64,sigma_s::Float64,t0::Float64,sigma_t::Float64)
 precmpcc=1e-3

 tb=(r,s,t)->-r
 prec_oracle=(r,s,t,prec)->max(min(r,s,t),prec) #max(r,s,t,prec)
 rho_restart(r,s,t,prec,rho)=rho
 paramin=sqrt(eps(Float64))
 #paramin=1.0
 initrst=()->(1.0,0.5,0.5)
 #to be defined: sigma_r,sigma_s,sigma_t
 updaterst=(r,s,t)->(0.1*r,0.1*s,0.05*t)

 ite_max_alas=100
 ite_max_viol=20
 rho_init=1*ones(nbc)
 rho_update=1.5 #2.0 bien pour Newton, trop grand sinon.
 rho_max=4000 #1/precmpcc #le grand max est 1/eps(Float64)
 goal_viol=0.75

 ite_max_armijo=100
 tau_armijo=0.1 #0.4
 armijo_update=0.9 #0.9
 ite_max_wolfe=10
 tau_wolfe=0.6
 wolfe_update=5.0

 verbose=0 #0 quiet, 1 relax, 2 relax+activation, 3 relax+activation+linesearch

 return ParamSet(precmpcc,tb,prec_oracle,
                 rho_restart,paramin,initrst,updaterst,
                 ite_max_alas,ite_max_viol,
                 rho_init,rho_update,rho_max,
                 goal_viol,
                 ite_max_armijo,tau_armijo,armijo_update,
                 ite_max_wolfe,tau_wolfe,wolfe_update,
                 verbose)
end

#end of module
end
