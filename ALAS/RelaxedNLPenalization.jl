#Functions that solves the non-linear relaxed program.
#Sélectionne et résoud le problème d'optimisation dans le sous domaine actif.
export RelaxedNLsolvePenalization

using ALASMPCCmod
using ActifMPCCmod

function RelaxedNLsolvePenalization(alas,ma::ActifMPCCmod.MPCC_actif,
                                    x0::Vector,
                                    feas::Float64,feasible,
                                    dual_feas,dual_feasible::Bool,
                                    gradpen::Vector,
                                    ht::Float64,
                                    dj::Vector,
                                    step::Float64,oa;verbose::Bool=true)
 rho=alas.rho_init
 l_max=ma.paramset.ite_max_viol
 rho_update=ma.paramset.rho_update
 obj_viol=ma.paramset.goal_viol

 # Initialisation multiplicateurs : uxl,uxu,ucl,ucu :
 uxl,uxu,ucl,ucu,usg,ush=LagrangeInit(alas,rho,x0)

 n=length(alas.mod.mp.meta.x0)-2*ma.nb_comp
 xjkl=x0
 l=0;Armijosuccess=true;small_step=false
 feask=feas
 gradpen_prec=gradpen

  #boucle pour augmenter le paramètre de pénalisation tant qu'on a pas baissé suffisament la violation
  while l<l_max && !dual_feasible && Armijosuccess && !small_step

   #Unconstrained Solver modifié pour résoudre le sous-problème avec contraintes de bornes
 xjkl,ma.w,dj,step,wnew,outputArmijo,small_step,ols,gradpen,ht=UnconstrainedMPCCActif.LineSearchSolve(ma,xjkl,dj,step,gradpen,ht)

   feas=MPCCmod.viol_contrainte_norm(alas.mod,xjkl);
   feasible=feas<=alas.prec

   dual_feas=norm(ActifMPCCmod.grad(ma,xjkl,gradpen),Inf)
   dual_feasible=dual_feas<=alas.mod.algoset.unconstrained_stopping(alas.prec,rho)

   OutputALASmod.Update(oa,xjkl,rho,feas,dual_feas,dj,step,ols,ht,n)

   Armijosuccess=(outputArmijo==0)
   if outputArmijo==2
    @show "Unbounded subproblem"
    return xj,EndingTest(alas,Armijosuccess,small_step,feas,dual_feas,k),rho,oa
   end
   l+=1
  end

  #On met à jour rho si ça n'a pas été:
  if (l==l_max || small_step || !Armijosuccess || dual_feasible) && (feas>obj_viol*feask && !feasible)
   verbose && print_with_color(:red, "Max ité Unc. Min. l=$l |x|=$(norm(xjkl,Inf)) |c(x)|=$feas |L'|=$dual_feas Arm=$Armijosuccess small_step=$small_step rho=$(norm(rho,Inf))  \n")
   rho=RhoUpdate(alas,rho,ma.crho,abs(MPCCmod.viol_contrainte(alas.mod,xjkl)))
   ActifMPCCmod.setcrho(ma,alas.mod.algoset.crho_update(feas,rho)) #on change le problème donc on réinitialise beta

   #met à jour la fonction objectif après la mise à jour de rho
   ma.nlp,ht,gradpen=UpdatePenaltyNLP(alas,rho,xjkl,usg,ush,uxl,uxu,ucl,ucu,ma.nlp,objpen=ht,gradpen=gradpen,crho=ma.crho)
   ActifMPCCmod.setbeta(ma,0.0) #on change le problème donc on réinitialise beta
   #ActifMPCCmod.sethess(ma,H) #on change le problème donc on réinitialise Hess
   dual_feas=norm(ActifMPCCmod.grad(ma,xjkl,gradpen),Inf)
   dual_feasible=dual_feas<=alas.prec
  end
  xjk=xjkl

  #Mise à jour des paramètres de la pénalité Lagrangienne (uxl,uxu,ucl,ucu)
  uxl,uxu,ucl,ucu,usg,ush=LagrangeUpdate(alas,rho,xjk,uxl,uxu,ucl,ucu,usg,ush)

  #met à jour la fonction objectif après la mise à jour des multiplicateurs
  temp,ht,gradpen=UpdatePenaltyNLP(alas,rho,xjk,usg,ush,uxl,uxu,ucl,ucu,
                                   ma.nlp,objpen=ht,gradpen=gradpen)

  multi_norm=1.0

  feasible=feas<=alas.prec
  dual_feasible=dual_feas/multi_norm<=alas.prec

 return ma,xjk,ht,gradpen,uxl,uxu,ucl,ucu,usg,ush,rho,Armijosuccess,small_step,feasible,dual_feasible
end
