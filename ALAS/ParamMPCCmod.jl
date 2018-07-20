module ParamMPCCmod

import SolveRelaxSubProblem.SolveSubproblemAlas

"""
Liste les paramètres utilisés dans la résolution du MPCC:
ParamSet(nbc::Int64)
ParamSet(nbc::Int64,r0::Float64,sigma_r::Float64,s0::Float64,sigma_s::Float64,t0::Float64,sigma_t::Float64)
"""

type ParamMPCC
 #paramètres pour la résolution du MPCC
 precmpcc :: Float64

 #paramètres algorithmiques pour la relaxation
 # l'oracle de choix de epsilon
 prec_oracle :: Function #dépend des trois paramètres (r,s,t)
 # la mise à jour des paramètres
 rho_restart :: Function #
 rho_init :: Vector
 paramin :: Float64 #valeur minimal pour les paramères (r,s,t)
 initrst :: Function #initialize (r,s,t)
 updaterst :: Function #update (r,s,t)

 #algo de relaxation:
 solve_sub_pb :: Function

 #paramètres de sortie:
 verbose::Int64
end

#Constructeur par défaut
#nbc est le nombre de contraintes du problème
function ParamMPCC(nbc::Int64)

 (sigma_r,sigma_s,sigma_t)=(0.1,0.1,0.05)
 (r0,s0,t0)=(1.0,0.5,0.5)
 return ParamMPCC(nbc,r0,sigma_r,s0,sigma_s,t0,sigma_t)
end

function ParamMPCC(nbc::Int64,
                  r0::Float64,sigma_r::Float64,
                  s0::Float64,sigma_s::Float64,
                  t0::Float64,sigma_t::Float64)

 precmpcc=1e-3

 prec_oracle=(r,s,t,prec)->max(min(r,s,t),prec) #max(r,s,t,prec)
 rho_restart(r,s,t,prec,rho)=rho
 rho_init=1*ones(nbc)
 paramin=sqrt(eps(Float64))
 #paramin=1.0
 initrst=()->(1.0,0.5,0.5)
 #to be defined: sigma_r,sigma_s,sigma_t
 updaterst=(r,s,t)->(0.1*r,0.1*s,0.05*t)

 solve_sub_pb = SolveSubproblemAlas

 verbose=0 #0 quiet, 1 relax, 2 relax+activation, 3 relax+activation+linesearch

 return ParamMPCC(precmpcc,prec_oracle,
                 rho_restart,rho_init,paramin,initrst,updaterst,
                 solve_sub_pb,
                 verbose)
end

#end of module
end
