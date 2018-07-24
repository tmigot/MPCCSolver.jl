module ParamSetmod

"""

Algorithmic parameters in the ALAS method

"""

type ParamSet

 tb             :: Function #dépend des trois paramètres (r,s,t)

 #active-set algorithmic parameters
 ite_max_alas   :: Int64
 ite_max_viol   :: Int64
 rho_init       :: Vector #un vecteur ?
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
 verbose        :: Int64
end

#Constructeur par défaut
#nbc: number of constraints of the problem
function ParamSet(nbc            :: Int64;
                  tb             :: Function = (r,s,t)->-r,
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
                  verbose        :: Int64    = 0 
                  )
 #@show nbc 
 #verbose: #0 quiet, 1 relax, 2 relax+activation, 3 relax+activation+linesearch

 return ParamSet(tb,ite_max_alas,ite_max_viol,
                 rho_init,rho_update,rho_max,
                 goal_viol,
                 ite_max_armijo,tau_armijo,armijo_update,
                 ite_max_wolfe,tau_wolfe,wolfe_update,
                 verbose)
end

#end of module
end
