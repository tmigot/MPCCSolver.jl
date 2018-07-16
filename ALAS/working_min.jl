function WorkingMin(alas :: ALASMPCC,
                    ma :: ActifMPCCmod.ActifMPCC,
                    xjk :: Vector,
                    rho :: Vector,
                    ht :: Float64,
                    gradpen :: Vector,
                    oa,
                    wnew,
                    step,
                    dj)

 n=alas.mod.n

  l=0
   ∇f=ActifMPCCmod.grad(ma,xjk,gradpen) #save one eval
  alas.sts,OK=start!(ma.nlp,alas.sts,xjk,∇f)

  subpb_fail=false
  #Boucle 1 : Etape(s) de minimisation dans le sous-espace de travail
  while !OK
   xjkl,ma.w,dj,step,wnew,outputArmijo,small_step,ols,gradpen,ht=LineSearchSolve(ma,xjk,
                                                                      dj,step,gradpen,ht)
   xjk=xjkl

   #Prochaine étape :
   #Faire sortir les contraintes actives du LineSearch

   Armijosuccess=(outputArmijo==0)
   alas.sts.unbounded=outputArmijo==2?true:false

     ∇f=ActifMPCCmod.grad(ma,xjk,gradpen)

   l+=1
   OK, elapsed_time=stop(ma.nlp,alas.sts,l,xjk,ht,∇f)
   
   OutputALASmod.Update(oa,xjk,rho,alas.spas.feasibility,alas.sts.optimality,dj,step,ols,ht,n)
   subpb_fail=!(Armijosuccess && !small_step)

   OK=OK && !subpb_fail
  end

 return xjk,alas,ma,dj,step,wnew,subpb_fail,gradpen,ht,l
end
