module ParamMPCCmod

import SolveRelaxSubProblem: solve_subproblem_rlx, solve_subproblem_IpOpt

type ParamMPCC

 #paramètres pour la résolution du MPCC
 precmpcc     :: Float64

 #paramètres algorithmiques pour la relaxation
 # l'oracle de choix de epsilon
 prec_oracle  :: Function #dépend des trois paramètres (r,s,t)

 # la mise à jour des paramètres
 rho_restart  :: Function #
 rho_init     :: Vector
 paramin      :: Float64 #valeur minimal pour les paramères (r,s,t)
 initrst      :: Function #initialize (r,s,t)
 updaterst    :: Function #update (r,s,t)

 #algo de relaxation:
 solve_sub_pb :: Function

 #paramètres de sortie:
 verbose      :: Int64

end

#Constructeur par défaut
#nbc est le nombre de contraintes du problème
function ParamMPCC(nbc          :: Int64;
                   precmpcc     :: Float64  = 1e-3,
                   prec_oracle  :: Function = (r,s,t,prec) -> max(min(r,s,t),prec),
                   rho_restart  :: Function = (r,s,t,rho) -> rho,
                   rho_init     :: Vector   = 1*ones(nbc),
                   paramin      :: Float64  = sqrt(eps(Float64)),
                   initrst      :: Function = ()->(1.0,0.5,0.5),
                   updaterst    :: Function = (r,s,t)->(0.1*r,0.1*s,0.05*t),
                   solve_sub_pb :: Function = solve_subproblem_rlx,
                   verbose      :: Int64    = 0)

 return ParamMPCC(precmpcc,
                  prec_oracle,
                  rho_restart,
                  rho_init,
                  paramin,
                  initrst,
                  updaterst,
                  solve_sub_pb,
                  verbose)
end

#end of module
end
