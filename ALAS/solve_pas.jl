export solvePas

function solvePAS(alas::ALASMPCC; verbose::Bool=true)

#Initialisation paramètres :
 n=alas.mod.n

 gradpen=Vector(n)
 gradpen_prec=Vector(n)
 xjk=Vector(n+2*alas.mod.nb_comp)
 xjkl=Vector(n+2*alas.mod.nb_comp)
 lambda=Vector(3*alas.mod.nb_comp)
 dj=zeros(n+2*alas.mod.nb_comp) #doit disparaitre

# S0 : initialisation du problème avec slack (projection sur _|_ )
 xjk=SlackComplementarityProjection(alas)

 ρ=alas.rho_init

 #variables globales en sortie du LineSearch
 wnew=zeros(Bool,0,0) #Tangi18: est-ce que ça chevauche le ma.wnew ?

 #Initialisation multiplicateurs de la contrainte de complémentarité
 lambda=LagrangeCompInit(alas,ρ,xjk) #améliorer si nb_comp=0.
 l_negative=findfirst(x->x<0,lambda)!=0

 # Initialisation multiplicateurs : uxl,uxu,ucl,ucu :
 uxl,uxu,ucl,ucu,usg,ush=LagrangeInit(alas,ρ,xjk)

 # S1 : Initialisation du ActifMPCC 
 ma=InitializeMPCCActif(alas,xjk,ρ,usg,ush,uxl,uxu,ucl,ucu)

 # S2 : Major Loop Activation de contrainte
 #initialisation
 ht,gradpen=NLPModels.objgrad(ma.nlp,xjk)
 ∇f=grad(ma,xjk,gradpen)

 alas.spas,GOOD = pas_start!(alas.mod,alas.spas,xjk,∇f,lambda)
 oa = OutputALAS(xjk,dj,alas.spas.feasibility,alas.spas.optimality,ρ,ht)

 #MAJOR LOOP
 k=0
 step=1.0;subpb_fail=false #pas besoin ici
 while !GOOD

  #Minization in the working set
  xjkl,alas,ma,dj,step,wnew,subpb_fail,gradpen,ht,l=WorkingMin(alas,ma,xjk,ρ,
                                                               ht,gradpen,oa,
                                                               wnew,step,dj)
  xjk=xjkl

  #Conditionnelle: met éventuellement rho à jour.
  alas.spas, UPDATE = pas_rhoupdate!(alas.mod,alas.spas,xjk)

   verbose && print_with_color(:red, "End - Min: l=$l |x|=$(norm(xjkl,Inf)) |c(x)|=$(alas.spas.feasibility) |L'|=$(alas.sts.optimality)  ρ=$(norm(ρ,Inf)) prec=$(alas.prec) \n")

  if UPDATE

   ρ,ma,ht,gradpen = CheckUpdateRho(alas,ma,xjk,ρ,alas.spas.feasibility,
                                    usg,ush,uxl,uxu,
                                    ucl,ucu,ht,gradpen,verbose)

  end

  #Mise à jour des paramètres de la pénalité Lagrangienne (uxl,uxu,ucl,ucu)
  uxl,uxu,ucl,ucu,usg,ush=LagrangeUpdate(alas,ρ,xjk,uxl,uxu,ucl,ucu,usg,ush)

  #met à jour la fonction objectif après la mise à jour des multiplicateurs
  ma.nlp,ht,gradpen=UpdatePenaltyNLP(alas,ρ,xjk,usg,ush,uxl,uxu,ucl,ucu,
                                   ma.nlp,objpen=ht,gradpen=gradpen) 
   ∇f=grad(ma,xjk,gradpen)

  #Mise à jour des multiplicateurs de la complémentarité
  if alas.mod.nb_comp>0 && alas.spas.wolfe_step
   lambda,alas.spas.l_negative=LSQComputationMultiplierBool(ma,gradpen,xjk)
  end

  #Relaxation rule:
  if (alas.spas.wolfe_step || step==0.0) && alas.spas.l_negative
  
   RelaxationRule(ma,xjk,lambda,wnew)

   verbose && print_with_color(:yellow, "Active set: $(ma.wcc) \n")

  end
  
  k+=1

  verbose && alas.spas.tired && print_with_color(:red, "Max ité. Lagrangien \n")

  #si on bloque mais qu'un multiplicateur est <0 on continue
  subpb_fail=subpb_fail && !alas.spas.l_negative 
  alas.spas,GOOD=pas_stop!(alas.mod,alas.spas,xjk,∇f,k,minimum(ρ))

  GOOD = alas.sts.unbounded || GOOD || subpb_fail

 end
 #MAJOR LOOP

 #Traitement finale :
 alas=addInitialPoint(alas,xjk)

 stat=ending_test(alas.spas,subpb_fail,alas.sts.unbounded)

 return xjk,stat,ρ,oa
end
