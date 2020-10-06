#import SolveRelaxSubProblem: solve_subproblem_rlx, solve_subproblem_IpOpt

"""

Algorithmic parameters in the ALAS method

"""
mutable struct ParamMPCC

 #paramètres pour la résolution du MPCC
 precmpcc       :: Float64

 #paramètres algorithmiques pour la relaxation
 # l'oracle de choix de epsilon
 prec_oracle    :: Function #dépend des trois paramètres (r,s,t)

 # la mise à jour des paramètres
 rho_restart    :: Function #
 paramin        :: Float64 #valeur minimal pour les paramères (r,s,t)
 initrst        :: Function #initialize (r,s,t)
 updaterst      :: Function #update (r,s,t)

 #algo de relaxation:
 solve_sub_pb   :: Function

 tb             :: Function #dépend des trois paramètres (r,s,t)

 #active-set algorithmic parameters
 ite_max_alas   :: Int64
 ite_max_viol   :: Int64
 rho_init       :: Vector
 rho_update     :: Float64 #nombre >=1
 rho_max        :: Float64
 goal_viol      :: Float64 #nombre entre (0,1)

 #paramètres pour la minimisation sans contrainte
 ite_max_armijo :: Int64 #nombre maximum d'itération pour la recherche linéaire >= 0
 tau_armijo     :: Float64 #paramètre pour critère d'Armijo doit être entre (0,0.5)
 armijo_update  :: Float64 #step=step_0*(1/2)^m
 ite_max_wolfe  :: Int64
 tau_wolfe      :: Float64 #entre (tau_armijo,1)
 wolfe_update   :: Float64

 #paramètres de sortie:
 verbose      :: Int64  #verbose: #0 quiet, 1 relax, 2 relax+activation, 3 relax+activation+linesearch

 uncmin        :: Function
 penalty       :: Function
 direction     :: Function
 linesearch    :: Function

 #Constructeur par défaut
 #nbc est le nombre de contraintes du problème
 function ParamMPCC(nbc            :: Int64;
                    precmpcc       :: Float64  = 1e-3,
                    prec_oracle    :: Function = (r,s,t,prec) -> max(min(r,s,t),prec),
                    rho_restart    :: Function = (r,s,t,rho) -> rho,
                    paramin        :: Float64  = sqrt(eps(Float64)),
                    initrst        :: Function = ()->(1.0,0.5,0.5),
                    updaterst      :: Function = (r,s,t)->(0.1*r,0.1*s,0.05*t),
                    solve_sub_pb   :: Function = x -> x,#solve_subproblem_rlx,
                    tb             :: Function = (r,s,t)->r,
                    ite_max_alas   :: Int64    = 100,
                    ite_max_viol   :: Int64    = 30,
                    rho_init       :: Vector   = ones(nbc),
                    rho_update     :: Float64  = 1.5,
                    rho_max        :: Float64  = 4000.0, #<=1/eps(Float64)
                    goal_viol      :: Float64  = 0.75,
                    ite_max_armijo :: Int64    = 100,
                    tau_armijo     :: Float64  = 0.1,
                    armijo_update  :: Float64  = 0.9,
                    ite_max_wolfe  :: Int64    = 10,
                    tau_wolfe      :: Float64  = 0.6,
                    wolfe_update   :: Float64  = 5.0,
                    verbose        :: Int64    = 0,
                    uncmin         :: Function = x -> x,#actif_solve, #working_min_proj, actif_solve
                    penalty        :: Function = x -> x,#Penalty.quadratic, #lagrangian, quadratic
                    direction      :: Function = x -> x,#DDirection.NwtdirectionSpectral, #NwtdirectionSpectral, CGHZ, invBFGS
                    linesearch     :: Function = x -> x)#LineSearch.armijo_wolfe, #armijo, armijo_wolfe, armijo_wolfe_hz)

  return new(precmpcc, prec_oracle, rho_restart, paramin,
             initrst, updaterst, solve_sub_pb, tb, ite_max_alas,ite_max_viol,
             rho_init,rho_update,rho_max, goal_viol, ite_max_armijo, tau_armijo,
             armijo_update, ite_max_wolfe,tau_wolfe,wolfe_update, verbose,
             uncmin, penalty, direction, linesearch)
 end
end
