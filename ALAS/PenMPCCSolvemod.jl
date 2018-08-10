module PenMPCCSolvemod

import ParamSetmod.ParamSet
import AlgoSetmod.AlgoSet

import RPenmod.RPen, RPenmod.pen_start!, RPenmod.pen_update!
import PenMPCCmod.PenMPCC
import StoppingPenmod.StoppingPen

import OutputALASmod.OutputALAS, OutputALASmod.oa_update!

import Relaxation.psi, Relaxation.dpsi, Relaxation.ddpsi, Relaxation.dphi

import StoppingPenmod.StoppingPen, StoppingPenmod.spen_start!
import StoppingPenmod.spen_stop!, StoppingPenmod.spen_final!

import Stopping.TStopping
import RActifmod.RActif, RActifmod.actif_start!

import ActifMPCCmod.ActifMPCC,ActifMPCCmod.grad
import ActifMPCCmod.setw
import ActifMPCCmod.lsq_computation_multiplier_bool, ActifMPCCmod.relaxation_rule!
import ActifMPCCmod.pas_max, ActifMPCCmod.redd
import ActifMPCCmod.evalx

importall NLPModels
#NLPModels.AbstractNLPModel,  NLPModelMeta, Counters

type PenMPCCSolve <: AbstractNLPModel

 meta       :: NLPModelMeta
 counters   :: Counters #ATTENTION : increment! ne marche pas?
 x0         :: Vector

 pen        :: PenMPCC

 w          :: Array{Bool,2} # n +2ncc x 2 matrix

 n          :: Int64 #dans le fond est optionnel si on a ncc
 ncc        :: Int64

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
 algoset    :: AlgoSet

 rpen       :: RPen
 spen       :: StoppingPen

end

############################################################################
#
#function ***
#
############################################################################

include("penmpccsolve.jl")

############################################################################
#
# Methods to update the PenMPCCSolve
#
############################################################################

#include("***")

############################################################################
#
# Classical NLP functions on ActifMPCC
# obj, grad, grad!, hess, cons, cons!
#
############################################################################

include("penmpccsolve_nlp.jl")

############################################################################
#
#function solve_subproblem_pen(ma      :: PenMPCCSolve,
#                              xjk     :: Vector,
#                              oa      :: OutputALAS;
#                              verbose :: Bool = true)
#
############################################################################

include("pen_solve.jl")

############################################################################
#
# calcul la valeur des multiplicateurs de Lagrange
#
#function _lsq_computation_multiplier_bool(ma  :: PenMPCCSolve,
#                                          xjk :: Vector) in n+2ncc
# output: lambda (Vector), l_negative (Bool)
#
#function _lsq_computation_multiplier(ma      :: PenMPCCSolve,
#                                     gradpen :: Vector,
#                                     xj      :: Vector)
# output: lambda (Vector)
############################################################################

include("penmpccsolve_compmultiplier.jl")

############################################################################
#
# Initialize ActifMPCC
#
# InitializeMPCCActif ***
# return: ActifMPCC
#
############################################################################

include("init_actifmpccsolve.jl")

#end of module
end
