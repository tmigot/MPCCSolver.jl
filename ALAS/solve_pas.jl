export solvePas

import RPenmod.pen_start!, RPenmod.pen_update!, RPenmod.pen_rho_update!
import PenMPCCmod.ComputationMultiplierBool
import PenMPCCmod.jac

function solvePAS(alas    :: ALASMPCC;
                  verbose :: Bool = true)

 n       = alas.mod.n
 nb_comp = alas.mod.nb_comp

 #SO : Projection of the initial point:
 xjk = SlackComplementarityProjection(alas)

 # S0: Initialize the parameters
 ρ      = alas.rho_init
 lambda = LagrangeCompInit(alas, ρ, xjk, c = alas.rrelax.feas_cc)
 u      = LagrangeInit(alas, ρ, xjk, c = alas.rrelax.feas)
 mult   = vcat(u[2*nb_comp+1:2*nb_comp+2*n],lambda)

 # S1 : Initialization of ActifMPCC 
 pen, rpen = InitializePenMPCC(alas, xjk, ρ, u) #create a new problem and result
 ma        = InitializeSolvePenMPCC(alas, pen, rpen, xjk)

 # S2 : Initialize Result and Stopping
 pen_start!(ma.pen, ma.rpen, xjk, lambda = mult)
 alas.spas, GOOD = pas_start!(alas.mod, alas.spas, xjk, ma.rpen)

 oa = OutputALAS(xjk, ma.dj, alas.spas.feasibility, 
                 alas.spas.optimality,ma.pen.ρ, ma.rpen.fx)

 #MAJOR LOOP
 k=0
 while !GOOD

  xjk, ma = solve_subproblem_pen(ma, xjk, oa)

  alas.spas, UPDATE = pas_rhoupdate!(alas.mod, alas.spas, xjk)

  ################ Update Penalty ############################################
  #Conditionnelle: met éventuellement rho à jour.

  if UPDATE

   ma.pen.ρ, ma = CheckUpdateRho(alas, ma, xjk, verbose)

  end

  #Mise à jour des paramètres de la pénalité Lagrangienne
  ma.pen.u = LagrangeUpdate(alas,ma.pen.ρ,xjk,ma.pen.u)

  #met à jour la fonction objectif après la mise à jour des multiplicateurs
  ma.pen.nlp, ma.rpen.fx, ma.rpen.gx = UpdatePenaltyNLP(alas, ma.pen.ρ, xjk,
                                                        ma.pen.u,
                                                        ma.pen.nlp,
                                                        objpen=ma.rpen.fx,
                                                        gradpen=ma.rpen.gx)

  ma.rpen.lambda, alas.spas.l_negative = LSQComputationMultiplierBool(ma,xjk) 
  #ne fait pas tout à fait la même chose
  #rpen.lambda,alas.spas.l_negative = ComputationMultiplierBool(pen,rpen.gx,xjk)

  pen_rho_update!(ma.pen, ma.rpen, xjk)
  ############ Fin Update Penalty ############################################

  k+=1

  verbose && alas.spas.tired && print_with_color(:red, "Max ité. Lagrangien \n")

  alas.spas,GOOD = pas_stop!(alas.mod,alas.spas,xjk,ma.rpen,k,minimum(ma.pen.ρ)) #k et rho ?

 end
 #MAJOR LOOP

 #Traitement finale :
 alas = set_x(alas, xjk)

 stat = ending_test(alas.spas, ma.rpen)

 return xjk, stat, ma.pen.ρ, oa
end

