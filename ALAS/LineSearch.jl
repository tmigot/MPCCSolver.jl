"""
Package de fonction pour la recherche linéaire d'un MPCCActif

liste des fonctions :
Armijo(ma::ActifMPCCmod.MPCC_actif,xj::Any,d::Any,hg::Any,old_grad::Any,stepmax::Float64,slope::Float64;
                verbose :: Bool=false, kwargs...)
ArmijoWolfe(ma::ActifMPCCmod.MPCC_actif,xj::Any,d::Any,hg::Any,old_grad::Any,stepmax::Float64,
                slope::Float64;verbose :: Bool=false, kwargs...)
"""
module LineSearch

using ActifMPCCmod

"""
Armijo : Backtracking line search
         1D Minimization
Amélioration : - changer armijo_update à chaque itération en garantissant rho in
               - utiliser le good_grad
"""
function Armijo(ma::ActifMPCCmod.MPCC_actif,xj::Any,d::Any,hg::Any,old_grad::Any,stepmax::Float64,slope::Float64;
                verbose :: Bool=false, kwargs...)

 good_grad=false
 nbW=0
 nbk=0
 #scale=norm(d)>=1000?norm(d):1.0
 scale=1.0
 dd=d/scale
 step=min(stepmax,1.0)*scale
 #slope = scale*dot(ActifMPCCmod.grad(ma,xj),d)

 #First try to increase t to satisfy loose Wolfe condition 
 ht=ActifMPCCmod.obj(ma,xj+step*d)
 slope_t=dot(d,ActifMPCCmod.grad(ma,xj+step*d))
 while (slope_t<ma.tau_wolfe*slope) && (ht-hg<=ma.tau_armijo*step*slope) && (nbW<ma.ite_max) && step<stepmax
  step=min(step*ma.wolfe_update,stepmax) #on ne peut pas dépasser le pas maximal
  ht=ActifMPCCmod.obj(ma,xj+step*d)
  slope_t=dot(d,ActifMPCCmod.grad(ma,xj+step*d))

  true && println("Wolfe Trick",nbW," ",hg," ",ht," ",ma.tau_armijo*step*slope)
  nbW+=1
 end

 ht=ActifMPCCmod.obj(ma,xj+step*dd)

 #critère d'Armijo : f(x+alpha*d)-f(x)<=tau_a*alpha*grad f^Td
 while nbk<ma.ite_max && ht-hg>ma.tau_armijo*step*slope
  step*=ma.armijo_update #step=step_0*(1/2)^m
  ht=ActifMPCCmod.obj(ma,xj+step*dd)
  nbk+=1
 end
 step=step/scale

 return step,good_grad,ht,nbk,nbW
end

"""
ArmijoWolfe : 'Newarmijo_wolfe' de JPD
Problème avec Hager et Zhang numerical trick
+ Ca n'a pas passé mon test... même sans Hager et Zhang trick...
"""
function ArmijoWolfe(ma::ActifMPCCmod.MPCC_actif,xj::Any,d::Any,hg::Any,old_grad::Any,stepmax::Float64,
                slope::Float64;verbose :: Bool=false, kwargs...)

 # Perform improved Armijo linesearch.
 nbk = 0
 nbW = 0
 step = min(stepmax,1.0)

 #First try to increase t to satisfy loose Wolfe condition 
 ht=ActifMPCCmod.obj(ma,xj+step*d)
 slope_t=dot(d,ActifMPCCmod.grad(ma,xj+step*d))
 while (slope_t<ma.tau_wolfe*slope) && (ht-hg<=ma.tau_armijo*step*slope) && (nbW<ma.ite_max) && step<stepmax
  step=min(step*ma.wolfe_update,stepmax) #on ne peut pas dépasser le pas maximal
  ht=ActifMPCCmod.obj(ma,xj+step*d)
  slope_t=dot(d,ActifMPCCmod.grad(ma,xj+step*d))

  true && println(nbW," ",hg," ",ht," ",ma.tau_armijo*step*slope)
  nbW+=1
 end

 ht=ActifMPCCmod.obj(ma,xj+step*d)
 hgoal = hg+slope*step*ma.tau_armijo
# Hager & Zhang numerical trick : pas certain...
 fact=-0.8
 #prec=1e-10
# Armijo = (ht <= hgoal) || ((ht <= hg + prec*abs(hg)) && (slope_t <= fact * slope))
 Armijo= (ht <= hgoal)
 good_grad=true
 #critère d'Armijo : f(x+alpha*d)-f(x)<=tau_a*alpha*grad f^Td
 while nbk<ma.ite_max && !Armijo
  step*=ma.armijo_update #step=step_0*(1/2)^m
  ht=ActifMPCCmod.obj(ma,xj+step*d)
  hgoal = hg+slope*step*ma.tau_armijo

  Armijo=false
  good_grad=false
  if ht <= hgoal
   Armijo=true
  elseif ht <= hg+1e-10*abs(hg)
   slope_t = dot(d,ActifMPCCmod.grad(ma,xj+step*d))
   good_grad=true
   if slope_t <= fact*slope
#    Armijo = true  #Hager and Zhan numerical trick
   end
  end

  nbk+=1
 end

 good_grad=false #à enlever ?
 return step,good_grad,ht,nbk,nbW
end

#end of module
end
