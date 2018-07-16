export solvePas


function solvePAS(alas::ALASMPCC; verbose::Bool=true)

#Initialisation paramètres :
 rho=alas.rho_init
 k_max=alas.paramset.ite_max_alas
 l_max=alas.paramset.ite_max_viol
 rho_update=alas.paramset.rho_update
 obj_viol=alas.paramset.goal_viol

 n=alas.mod.n

 gradpen=Vector(n)
 gradpen_prec=Vector(n)
 xjk=Vector(n+2*alas.mod.nb_comp)
 l=Vector(3*alas.mod.nb_comp)

# S0 : initialisation du problème avec slack (projection sur _|_ )
 xjk=SlackComplementarityProjection(alas)

 #variables globales en sortie du LineSearch
 wnew=zeros(Bool,0,0) #Tangi18: est-ce que ça chevauche le ma.wnew ?

 #Initialisation multiplicateurs de la contrainte de complémentarité
 lambda=LagrangeCompInit(alas,rho,xjk) #améliorer si nb_comp=0.
 l_negative=findfirst(x->x<0,lambda)!=0

 # Initialisation multiplicateurs : uxl,uxu,ucl,ucu :
 uxl,uxu,ucl,ucu,usg,ush=LagrangeInit(alas,rho,xjk)

 # S1 : Initialisation du ActifMPCC 
 ma=InitializeMPCCActif(alas,xjk,rho,usg,ush,uxl,uxu,ucl,ucu)

 # S2 : Major Loop Activation de contrainte
 #initialisation
 ht,gradpen=NLPModels.objgrad(ma.nlp,xjk)
 ∇f=ActifMPCCmod.grad(ma,xjk,gradpen)

 #direction initial
 dj=zeros(n+2*alas.mod.nb_comp)

 alas.spas,GOOD=pas_start!(alas.mod,alas.spas,xjk,∇f,lambda)
 oa=OutputALASmod.OutputALAS(xjk,dj,alas.spas.feasibility,alas.spas.optimality,rho,ht)
 #MAJOR LOOP
 k=0
 step=1.0;Armijosuccess=true;small_step=false #pas besoin ici
 while !GOOD

  gradpen_prec=gradpen

  #Minization in the working set
  xjkl,alas,ma,dj,step,wnew,subpb_fail,gradpen,ht,l=WorkingMin(alas,ma,xjk,rho,ht,gradpen,oa,wnew,step,dj)
  xjk=xjkl

  #devrait être tout à la fin ?
  if alas.sts.unbounded
   @show "Unbounded subproblem"
   return xj,EndingTest(alas,Armijosuccess,small_step,alas.spas.feasibility,alas.sts.optimality,k),rho,oa
  end
  ############################

  #Conditionnelle: met éventuellement rho à jour.
  alas.spas, UPDATE = pas_rhoupdate!(alas.mod,alas.spas,xjk)
  if UPDATE

   verbose && print_with_color(:red, "Max ité Unc. Min. l=$l |x|=$(norm(xjkl,Inf)) |c(x)|=$(alas.spas.feasibility) |L'|=$(alas.sts.optimality) Arm=$Armijosuccess small_step=$small_step rho=$(norm(rho,Inf))  \n")

   rho,ma,ht,gradpen = CheckUpdateRho(alas,ma,xjk,rho,alas.spas.feasibility,
                                    usg,ush,uxl,uxu,
                                    ucl,ucu,ht,gradpen,verbose)

  end

  #Mise à jour des paramètres de la pénalité Lagrangienne (uxl,uxu,ucl,ucu)
  uxl,uxu,ucl,ucu,usg,ush=LagrangeUpdate(alas,rho,xjk,uxl,uxu,ucl,ucu,usg,ush)

  #met à jour la fonction objectif après la mise à jour des multiplicateurs
  ma.nlp,ht,gradpen=UpdatePenaltyNLP(alas,rho,xjk,usg,ush,uxl,uxu,ucl,ucu,
                                   ma.nlp,objpen=ht,gradpen=gradpen) 
   ∇f=ActifMPCCmod.grad(ma,xjk,gradpen)

  #Mise à jour des multiplicateurs de la complémentarité
  if alas.mod.nb_comp>0
   lambda,alas.spas.l_negative=ActifMPCCmod.LSQComputationMultiplierBool(ma,gradpen,xjk)
  end

  #Relaxation rule si on a fait un pas de Armijo-Wolfe et si un multiplicateur est du mauvais signe
  alas.spas.wolfe_step=dot(gradpen,dj)>=alas.paramset.tau_wolfe*dot(gradpen_prec,dj)
  #if k!=0 && (alas.spas.wolfe_step || step==0.0) && alas.spas.l_negative
  if (alas.spas.wolfe_step || step==0.0) && alas.spas.l_negative
  
   ActifMPCCmod.RelaxationRule(ma,xjk,lambda,wnew)
   verbose && print_with_color(:yellow, "Active set: $(ma.wcc) \n")

  end
  
  k+=1
  verbose && k>=k_max && print_with_color(:red, "Max ité. Lagrangien \n")

  #si on bloque mais qu'un multiplicateur est <0 on continue
  subpb_fail=subpb_fail || alas.spas.l_negative 
  alas.spas,GOOD=pas_stop!(alas.mod,alas.spas,xjk,∇f,k,minimum(rho))
  GOOD = GOOD && !subpb_fail

 end
 #MAJOR LOOP

 #Traitement finale :
 alas=addInitialPoint(alas,xjk)

 stat=EndingTest(alas,Armijosuccess,small_step,alas.spas.feasibility,alas.spas.optimality,k)

 return xjk,stat,rho,oa
end
