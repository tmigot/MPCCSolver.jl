module ActifMPCCmod

import ParamSetmod.ParamSet

import RPenmod.RPen
import PenMPCCmod.PenMPCC
import Stopping.TStopping

import OutputALASmod.oa_update!

import Relaxation.psi, Relaxation.dpsi, Relaxation.ddpsi, Relaxation.dphi

importall NLPModels
#NLPModels.AbstractNLPModel,  NLPModelMeta, Counters

type ActifMPCC <: AbstractNLPModel

 meta       :: NLPModelMeta
 counters   :: Counters #ATTENTION : increment! ne marche pas?
 x0         :: Vector

 pen        :: PenMPCC

 w          :: Array{Bool,2} # n +2ncc x 2 matrix

 n          :: Int64 #dans le fond est optionnel si on a ncc
 ncc    :: Int64

 #set of indices:
 wnc        :: Array{Int64,1} # indices of free variables in 1,...,n
 wn1        :: Array{Int64,1} # indices with active lower bounds in 1,...,n
 wn2        :: Array{Int64,1} # indices with active upper bounds in 1,...,n

 #ensembles d'indices complementarity:
 # ensemble des indices (entre 0 et ncc) où la contrainte yG>=-r est active
 w1         :: Array{Int64,1}
 # ensemble des indices (entre 0 et ncc) où la contrainte yH>=-r est active
 w2         :: Array{Int64,1} 
 # ensemble des indices (entre 0 et ncc) où la contrainte yG<=s+t*theta(yH,r) est active
 w3         :: Array{Int64,1}
 # ensemble des indices (entre 0 et ncc) où la contrainte yH<=s+t*theta(yG,r) est active
 w4         :: Array{Int64,1} 
 #ensemble des indices (entre 0 et ncc) où la contrainte Phi<=0 est active
 wcomp      :: Array{Int64,1}
 w13c       :: Array{Int64,1} #ensemble des indices où les variables yG sont libres
 w24c       :: Array{Int64,1} #ensemble des indices où les variables yH sont libres
 #ensemble des indices des contraintes où yG et yH sont libres
 wc         :: Array{Int64,1}
 #ensemble des indices des contraintes où yG et yH sont fixés
 wcc        :: Array{Int64,1} 


 wnew       :: Array{Bool,2} #dernières contraintes ajoutés
 dj         :: Vector #previous direction

 #paramètres pour le calcul de la direction de descente
 crho       :: Float64 #constant such that : ||c(x)||_2 \approx crho*rho
 beta       :: Float64 #paramètre pour gradient conjugué
 Hess       :: Array{Float64,2} #inverse matrice hessienne approximée
 #Hd       ::Vector #produit inverse matrice hessienne et gradient (au lieu de la hessienne entière)

 paramset   :: ParamSet
 direction  :: Function #fonction qui calcul la direction de descente
 linesearch :: Function #fonction qui calcul la recherche linéaire

 rpen       :: RPen
 sts        :: TStopping

end

############################################################################
#
#function ActifMPCC(pen        :: PenMPCC,
#                   ncc    :: Int64,
#                   paramset   :: ParamSet,
#                   direction  :: Function,
#                   linesearch :: Function,
#                   sts        :: TStopping,
#                   rpen       :: RPen)
#
############################################################################

include("actifmpcc.jl")

############################################################################
#
# Methods to update the ActifMPCC
# setw, updatew, setbeta, setcrho, sethess
#
############################################################################

include("actifmpcc_setteur.jl")

############################################################################
#
# Functions to switch from working space to whole space
# exalx, evald, redd
#
############################################################################

include("actifmpcc_eval.jl")

############################################################################
#
# Classical NLP functions on ActifMPCC
# obj, grad, grad!, hess, cons, cons!
#
############################################################################

include("actifmpcc_nlp.jl")

############################################################################
#
#function solve_subproblem_pen(ma      :: ActifMPCC,
#                              xjk     :: Vector,
#                              oa      :: OutputALAS;
#                              verbose :: Bool = true)
#
############################################################################
import OutputALASmod.OutputALAS
import RActifmod.RActif, RActifmod.actif_start!
import Stopping.TStopping, Stopping.start!, Stopping.stop
import RPenmod.RPen, RPenmod.pen_start!, RPenmod.pen_update!
import PenMPCCmod.PenMPCC

include("solve_subproblem_pen.jl")

############################################################################
#
# Minimisation sans contrainte dans domaine actif
#
############################################################################

include("working_min.jl")

import OutputLSmod.OutputLS

include("unconstrained_actifmpcc.jl")

############################################################################
#
# calcul la valeur des multiplicateurs de Lagrange
#
#function _lsq_computation_multiplier_bool(ma  :: ActifMPCC,
#                                          xjk :: Vector) in n+2ncc
# output: lambda (Vector), l_negative (Bool)
#
#function _lsq_computation_multiplier(ma      :: ActifMPCC,
#                                     gradpen :: Vector,
#                                     xj      :: Vector)
# output: lambda (Vector)
############################################################################

include("actifmpcc_compmultiplier.jl")

############################################################################
#
# Define the relaxation rule over the active constraints
#
#function relaxation_rule!(ma :: ActifMPCC,
#                          xj :: Vector,
#                          l :: Vector,
#                          wmax :: Array{Bool,2})
#
# Input:
#     xj          : vector of size n+2ncc
#     l           : vector of multipliers size (n + 2ncc)
#     wmax        : set of freshly added constraints
#
#return: ActifMPCC (with updated active constraints)
#
############################################################################

function relaxation_rule!(ma   :: ActifMPCC,
                          xj   :: Vector,
                          l    :: Vector,
                          wmax :: Array{Bool,2})

  copy_wmax = copy(wmax)

  n   = ma.n
  ncc = ma.ncc

  llx  = l[1:n]
  lux  = l[n+1:2*n]
  lg   = l[2*n+1:2*n+ncc]
  lh   = l[2*n+ncc+1:2*n+2*ncc]
  lphi = l[2*n+2*ncc+1:2*n+3*ncc]

  # Relaxation de l'ensemble d'activation : 
  # désactive toutes les contraintes négatives
  ll = [llx; lg; lphi; lux; lh; lphi] #pas très catholique comme technique
  ma.w[find(x -> x<0, ll)] = zeros(Bool, length(find(x -> x<0, ll)))

  # Règle d'anti-cyclage : 
  # on enlève pas une contrainte qui vient d'être ajouté.
  ma.w[find(x->x==1.0, copy_wmax)] = ones(Bool,length(find(x->x==1.0, copy_wmax)))

 return updatew(ma)
end

############################################################################
#
# Compute the maximum step to stay feasible in a direction
#
# pas_max(ma::ActifMPCC,x::Vector,d::Vector)
#
# return: alpha, (the step)
#         w_save, (contraintes actives au point x+step*d)
#         w_new  (les nouvelles contraintes actives au point x+step*d)
#
############################################################################
#- copie de code dans PasMax
import Relaxation.invpsi, Relaxation.alpha_theta_max

include("actifmpcc_pasmax.jl")

#end of module
end
