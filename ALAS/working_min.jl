importall LSDescentMethods

function WorkingMin(alas :: ALASMPCC,
                    ma :: ActifMPCC,
                    xjk :: Vector,
                    ρ :: Vector,
                    ht :: Float64,
                    gradpen :: Vector,
                    oa,
                    wnew,
                    step,
                    dj)

 n=alas.mod.n
 gradpen_prec=copy(gradpen) #sert juste pour checker wolfe (doit disparaitre)

 l=0
  ∇f=grad(ma,xjk,gradpen) #one eval can be saved

 alas.sts,OK=start!(ma,alas.sts,xjk,∇f)

 subpb_fail=false
 #Boucle 1 : Etape(s) de minimisation dans le sous-espace de travail
 while !OK

  xjkl,ma.w,dj,step,wnew,outputArmijo,small_step,ols,gradpen,ht=LineSearchSolve(ma,
                                                                               xjk,
                                                                                dj,
                                                                              step,
                                                                           gradpen,
                                                                                ht)
  xjk=xjkl

  #Prochaine étape :
  #Faire sortir les contraintes actives du LineSearch

  Armijosuccess = (outputArmijo==0)
  alas.sts.unbounded = outputArmijo==2 ? true : false

    ∇f = grad(ma,xjk,gradpen)

  l+=1
  OK, elapsed_time = stop(ma,alas.sts,l,xjk,ht,∇f)
   
  Update(oa,xjk,ρ,alas.spas.feasibility,alas.sts.optimality,dj,step,ols,ht,n)

  subpb_fail=!(Armijosuccess && !small_step)

  OK = OK && !subpb_fail
 end

alas.spas.wolfe_step=dot(gradpen,dj)>=alas.paramset.tau_wolfe*dot(gradpen_prec,dj)

 return xjk,alas,ma,dj,step,wnew,subpb_fail,gradpen,ht,l
end

###################################################################################
#
#
#
###################################################################################

importall LSDescentMethods
import ActifMPCCmod.evalx, ActifMPCCmod.PasMax, ActifMPCCmod.setw

function WorkingMinProj(alas :: ALASMPCC,
                    ma :: ActifMPCC,
                    xjk :: Vector,
                    ρ :: Vector,
                    ht :: Float64,
                    gradpen :: Vector,
                    oa,
                    wnew,
                    step,
                    dj)

 n=alas.mod.n
 ∇f=grad(ma,xjk,gradpen)

 #alas.sts,OK=start!(ma,alas.sts,xjk,∇f)
 x_old = ma.x0
 #2) Newton qui rend les 2 derniers itérés
  (x,d, f, tmp, iter,
  optimal, tired, status,   ma.counters.neval_obj,
  ma.counters.neval_grad, ma.counters.neval_hess)=Newton(ma,
                                                         x0=ma.x0,
                                                         stp=alas.sts,
                                     Nwtdirection=NwtdirectionLDLt)
#NwtdirectionLDLt, NwtdirectionSpectral, CG_HZ
 if !tired && !alas.sts.actfeas   #backtracking:
   #Calcul du pas maximum:
   step,wmax,wnew = PasMax(ma,x_old,d)
   x = x_old + step*d
   xjk = evalx(ma,x)
   ma.x0 = xjk
   #We hit a new constraint:
   setw(ma,wmax)
   alas.spas.wolfe_step = false
 elseif !tired
   ma.x0 = x
   xjk = evalx(ma,x)
   alas.spas.wolfe_step = true
 end

 ht,gradpen = objgrad(ma,xjk)
 subpb_fail = tired

 return xjk,alas,ma,d,step,wnew,subpb_fail,gradpen,ht,iter
end

#include("tmp.jl")
