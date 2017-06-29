module ALASMPCCmod

using OutputALASmod
using ActifMPCCmod
using UnconstrainedMPCCActif
using MPCCmod
using Relaxation

using NLPModels

"""
Type ALASMPCC : 

liste des constructeurs :
ALASMPCC(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64, prec::Float64)
ALASMPCC(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64, prec::Float64, rho::Vector)

liste des méthodes :


liste des accesseurs :


liste des fonctions :
solvePAS(alas::ALASMPCC)

EndingTest(alas::ALASMPCC,Armijosuccess::Bool,small_step::Bool,feas::Float64,dual_feas::Float64,k::Int64)
SlackComplementarityProjection(alas::ALASMPCC)
LagrangeInit(alas::ALASMPCC,rho::Vector,xj::Vector)
LagrangeCompInit(alas::ALASMPCC,rho::Vector,xj::Vector)
LagrangeUpdate(alas::ALASMPCC,rho::Vector,xjk::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector,usg::Vector,ush::Vector)
Penaltygen(alas::ALASMPCCmod.ALASMPCC,x::Vector,yg::Vector,yh::Vector,rho::Vector,usg::Vector,ush::Vector,uxl::Vector,uxu::Vector,ucl::Vector,ucu::Vector)
RhoUpdate(rho::Vector,err::Vector,alas::ALASMPCC)
RhoDetail(rho::Vector,alas::ALASMPCC)
"""

type ALASMPCC
 mod::MPCCmod.MPCC
 #paramètres du pb
 r::Float64
 s::Float64
 t::Float64
 tb::Float64

 #paramètres algorithmiques
 prec::Float64 #precision à 0
 rho_init::Vector #nombre >= 0
end

function ALASMPCC(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64, prec::Float64)
 rho_init=mod.paramset.rho_init
 return ALASMPCC(mod,r,s,t,prec,rho_init)
end

function ALASMPCC(mod::MPCCmod.MPCC,r::Float64,s::Float64,t::Float64, prec::Float64, rho::Vector)

 rho_init=rho
 tb=mod.paramset.tb(r,s,t)

 return ALASMPCC(mod,r,s,t,tb,prec,rho_init)
end

"""
solvePAS() : méthode de penalisation/activation de contraintes DHKM 16'

Méthode avec gestion à l'intérieur du paramètre de pénalité
"""
function solvePAS(alas::ALASMPCC; verbose::Bool=false)

#Initialisation paramètres :
 rho=alas.rho_init
 k_max=alas.mod.paramset.ite_max_alas
 l_max=alas.mod.paramset.ite_max_viol
 rho_update=alas.mod.paramset.rho_update
 obj_viol=alas.mod.paramset.goal_viol

 n=length(alas.mod.mp.meta.x0)

 gradpen=Vector(n)
 gradpen_prec=Vector(n)
 xj=Vector(n+2*alas.mod.nb_comp)
 xjk=Vector(n+2*alas.mod.nb_comp)
 xjkl=Vector(n+2*alas.mod.nb_comp)

# S0 : initialisation du problème avec slack (projection sur _|_ )
 xj=SlackComplementarityProjection(alas)

 #variables globales en sortie du LineSearch
 step=1.0
 Armijosuccess=true
 wnew=[]

 #Initialisation multiplicateurs de la contrainte de complémentarité
 lg,lh,lphi=LagrangeCompInit(alas,rho,xj)

 # Initialisation multiplicateurs : uxl,uxu,ucl,ucu :
 uxl,uxu,ucl,ucu,usg,ush=LagrangeInit(alas,rho,xj)

# S1 : Initialisation du MPCC_Actif 
 #objectif du problème avec pénalité lagrangienne sans contrainte :
 pen_mpcc=CreatePenaltyNLP(alas,xj,rho,usg,ush,uxl,uxu,ucl,ucu)

 #initialise notre problème pénalisé avec des contraintes actives w
 ma=ActifMPCCmod.MPCC_actif(pen_mpcc,alas.r,alas.s,alas.t,alas.mod.nb_comp,alas.mod.paramset,alas.mod.algoset.direction,alas.mod.algoset.linesearch)

 # S2 : Major Loop Activation de contrainte
 #initialisation
 k=0
 xjk=xj
 small_step=false
 gradpen=NLPModels.grad(ma.nlp,xj)

 #direction initial
 dj=zeros(n+2*alas.mod.nb_comp)

 l_negative=findfirst(x->x<0,[lg;lh;lphi])!=0

 feas=MPCCmod.viol_contrainte_norm(alas.mod,xjk)

 feasible=feas<=alas.prec
 dual_feas=norm(gradpen[1:n],Inf)
 multi_norm=alas.mod.algoset.scaling_dual(usg,ush,uxl,uxu,ucl,ucu,lg,lh,lphi,alas.prec,alas.mod.paramset.precmpcc,norm(rho,Inf),dual_feas)
 dual_feasible=dual_feas/multi_norm<=alas.prec

 oa=OutputALASmod.OutputALAS(xjk,dj,feas,dual_feas,rho)

 while k<k_max && (l_negative || !feasible || !dual_feasible) && Armijosuccess && !small_step

  l=0
  feask=feas
  gradpen_prec=gradpen

  #boucle pour augmenter le paramètre de pénalisation tant qu'on a pas baissé suffisament la violation
  while l<l_max && (l==0 || (k!=0 && feas>obj_viol*feask && !feasible )) && Armijosuccess && !small_step

   #On ne met pas à jour rho si tout se passe bien.
   if l!=0
    rho=RhoUpdate(alas,rho,abs(MPCCmod.viol_contrainte(alas.mod,xjkl)))
    #met à jouer la fonction objectif après la mise à jour de rho
    ma.nlp=UpdatePenaltyNLP(alas,xjk,rho,usg,ush,uxl,uxu,ucl,ucu,ma.nlp)
    gradpen=NLPModels.grad(ma.nlp,xjk) #intégrer dans "UpdatePenaltyNLP" pour éviter ce calcul
   end

   #Unconstrained Solver modifié pour résoudre le sous-problème avec contraintes de bornes
   xjkl,ma.w,dj,step,wnew,outputArmijo,small_step,ols,gradpen=UnconstrainedMPCCActif.LineSearchSolve(ma,xjk,dj,step,gradpen)

   feas=MPCCmod.viol_contrainte_norm(alas.mod,xjkl)
   dual_feas=norm(gradpen[1:n],Inf) #(n'est pas utile ici)

   OutputALASmod.Update(oa,xjkl,rho,feas,dual_feas,dj,step,ols)

   Armijosuccess=(outputArmijo==0)
   l+=1
   verbose && l==l_max && print_with_color(:red, "Max itération en l (rho update). Iteration: $k ||rho||=$(norm(rho,Inf))\n")
  end
  xjk=xjkl

  #Mise à jour des paramètres de la pénalité Lagrangienne (uxl,uxu,ucl,ucu)
  uxl,uxu,ucl,ucu,usg,ush=LagrangeUpdate(alas,rho,xjk,uxl,uxu,ucl,ucu,usg,ush)

  #met à jouer la fonction objectif après la mise à jour des multiplicateurs
  ma.nlp=UpdatePenaltyNLP(alas,xjk,rho,usg,ush,uxl,uxu,ucl,ucu,ma.nlp)
  gradpen=NLPModels.grad(ma.nlp,xjk) #intégrer dans "UpdatePenaltyNLP" pour éviter ce calcul

  #Mise à jour des multiplicateurs de la complémentarité
  if alas.mod.nb_comp>0
   lg,lh,lphi=ActifMPCCmod.LSQComputationMultiplier(ma,gradpen,xjk)
   l_negative=findfirst(x->x<0,[lg;lh;lphi])!=0 #vrai si un multiplicateur est negatif
  end
  multi_norm=alas.mod.algoset.scaling_dual(usg,ush,uxl,uxu,ucl,ucu,lg,lh,lphi,alas.prec,alas.mod.paramset.precmpcc,norm(rho,Inf),dual_feas)

  feasible=feas<=alas.prec
  dual_feasible=dual_feas/multi_norm<=alas.prec

  if l_negative && (!Armijosuccess || small_step) #si on bloque mais qu'un multiplicateur est <0 on continue
   Armijosuccess=true
   small_step=false
  end

  #Relaxation rule si on a fait un pas de Armijo-Wolfe et si un multiplicateur est du mauvais signe
  #quoi faire si on est bloqué ? i.e pas=0.0, on tente de relaxer
  if k!=0 && (dot(gradpen,dj)>=alas.mod.paramset.tau_wolfe*dot(gradpen_prec,dj) || step==0.0) && findfirst(x->x<0,[lg;lh;lphi])!=0
   ActifMPCCmod.RelaxationRule(ma,xjk,lg,lh,lphi,wnew)
  end

  k+=1
  k>=k_max && print_with_color(:red, "Max itération Lagrangien augmenté\n")
 end

#Traitement finale :
 xj=xjk

 stat=EndingTest(alas,Armijosuccess,small_step,feas,dual_feas,k)

 return xj,stat,rho,oa;
end

function EndingTest(alas::ALASMPCC,Armijosuccess::Bool,small_step::Bool,feas::Float64,dual_feas::Float64,k::Int64)

 stat=0
 if !Armijosuccess
  print_with_color(:red, "Failure : Armijo Failure\n")
  stat=1
 end
 if small_step
  print_with_color(:red, "Failure : too small step\n")
  #stat=1
 end
 if feas>alas.prec
  print_with_color(:red, "Failure : Infeasible Solution. norm: $feas\n")
  #stat=1
 end
 if dual_feas>alas.prec
  if k>=alas.mod.paramset.ite_max_alas
   print_with_color(:red, "Failure : Non-optimal Sol. norm: $dual_feas\n")
   stat=1
  else
   print_with_color(:red, "Inexact : Fritz-John Sol. norm: $dual_feas\n")
   #stat=0
  end
 end

 return stat
end

"""
SlackComplementarityProjection : 
projete le point x0 sur la contrainte de complémentarité avec les slack
pré-requis : x0 doit être de taille n+2q
"""
function SlackComplementarityProjection(alas::ALASMPCC)

 nb_comp=alas.mod.nb_comp

 if nb_comp==0
  return alas.mod.xj
 end

 #initialisation :
 #x=[alas.mod.xj;alas.mod.G(alas.mod.xj);alas.mod.H(alas.mod.xj)]
 x=[alas.mod.xj;NLPModels.cons(alas.mod.G,alas.mod.xj);NLPModels.cons(alas.mod.H,alas.mod.xj)]
 n=length(x)-2*nb_comp

 #projection sur les contraintes de positivité relaxés : yG>=tb et yH>=tb
 x[n+1:n+nb_comp]=max(x[n+1:n+nb_comp],ones(nb_comp)*alas.tb)
 x[n+nb_comp+1:n+2*nb_comp]=max(x[n+nb_comp+1:n+2*nb_comp],ones(nb_comp)*alas.tb)

 #projection sur les contraintes papillons : yG<=psi(yH,r,s,t) et yH<=psi(yG,r,s,t)
 for i=1:nb_comp
  psiyg=Relaxation.psi(x[n+i],alas.r,alas.s,alas.t)
  psiyh=Relaxation.psi(x[n+nb_comp+i],alas.r,alas.s,alas.t)

  if x[n+i]-psiyh>0 && x[n+nb_comp+i]-psiyg>0
   x[n+i]>=x[n+nb_comp+i] ? x[n+nb_comp+i]=psiyg : x[n+i]=psiyh
  end
 end

 return x
end

"""
Initialisation des multiplicateurs de Lagrange
"""
function LagrangeInit(alas::ALASMPCC,rho::Vector,xj::Vector)

  n=length(alas.mod.mp.meta.x0)
  rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=RhoDetail(alas,rho)

  uxl=max(rho_ineq_lvar.*(alas.mod.mp.meta.lvar-xj[1:n]),zeros(n))
  uxu=max(rho_ineq_uvar.*(xj[1:n]-alas.mod.mp.meta.uvar),zeros(n))
  if alas.mod.mp.meta.ncon!=0
   c(z)=NLPModels.cons(alas.mod.mp,z)
   nc=length(alas.mod.mp.meta.y0) #nombre de contraintes

   ucl=max(rho_ineq_lcons.*(alas.mod.mp.meta.lcon-c(xj[1:n])),zeros(nc))
   ucu=max(rho_ineq_ucons.*(c(xj[1:n])-alas.mod.mp.meta.ucon),zeros(nc))
  else
   ucl=[];ucu=[];
  end

  usg=zeros(alas.mod.nb_comp)
  ush=zeros(alas.mod.nb_comp)

 return uxl,uxu,ucl,ucu,usg,ush
end

function LagrangeCompInit(alas::ALASMPCC,rho::Vector,xj::Vector)

 n=length(alas.mod.mp.meta.x0)
 nb_comp=alas.mod.nb_comp
 rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=RhoDetail(alas,rho)
 
 psiyg=Relaxation.psi(xj[n+1:n+nb_comp],alas.r,alas.s,alas.t)
 psiyh=Relaxation.psi(xj[n+1+nb_comp:n+2*nb_comp],alas.r,alas.s,alas.t)
 phi=(xj[n+1:n+nb_comp]-psiyh).*(xj[n+1+nb_comp:n+2*nb_comp]-psiyg)

 lphi=max(phi,zeros(nb_comp))
 lg=max(-rho_eqg.*(xj[n+1:n+nb_comp]-alas.tb),zeros(nb_comp))
 lh=max(-rho_eqh.*(xj[n+1+nb_comp:n+2*nb_comp]-alas.tb),zeros(nb_comp))

 return lg,lh,lphi
end

"""
Mise à jour des multiplicateurs de Lagrange
"""
function LagrangeUpdate(alas::ALASMPCC,rho::Vector,xjk::Vector,
                        uxl::Vector,uxu::Vector,
                        ucl::Vector,ucu::Vector,
                        usg::Vector,ush::Vector)

  n=length(alas.mod.mp.meta.x0)
   rho_eqg,rho_eqh,rho_ineq_lvar,rho_ineq_uvar,rho_ineq_lcons,rho_ineq_ucons=RhoDetail(alas,rho)

  uxl=uxl+max(rho_ineq_lvar.*(xjk[1:n]-alas.mod.mp.meta.uvar),-uxl)
  uxu=uxu+max(rho_ineq_uvar.*(alas.mod.mp.meta.lvar-xjk[1:n]),-uxu)
  if alas.mod.mp.meta.ncon!=0
   c(z)=NLPModels.cons(alas.mod.mp,z)
   ucl=ucl+max(rho_ineq_lcons.*(c(xjk[1:n])-alas.mod.mp.meta.ucon),-ucl)
   ucu=ucu+max(rho_ineq_ucons.*(alas.mod.mp.meta.lcon-c(xjk[1:n])),-ucu)
  end

  if alas.mod.nb_comp!=0
   G(x)=NLPModels.cons(alas.mod.G,x)
   H(x)=NLPModels.cons(alas.mod.H,x)
   usg=usg+rho_eqg.*(G(xjk[1:n])-xjk[n+1:n+alas.mod.nb_comp])
   ush=ush+rho_eqh.*(H(xjk[1:n])-xjk[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])
  end

 return uxl,uxu,ucl,ucu,usg,ush
end


"""
Fonction de pénalité générique :
"""
function Penaltygen(alas::ALASMPCCmod.ALASMPCC,
                    x::Vector,yg::Vector,yh::Vector,
                    rho::Vector,usg::Vector,ush::Vector,
                    uxl::Vector,uxu::Vector,
                    ucl::Vector,ucu::Vector)

 if alas.mod.nb_comp>0
  G(x)=NLPModels.cons(alas.mod.G,x)
  H(x)=NLPModels.cons(alas.mod.H,x)
  return alas.mod.algoset.penalty(alas.mod.mp,G,H,alas.mod.nb_comp,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 else
  return alas.mod.algoset.penalty(alas.mod.mp,x->(),x->(),alas.mod.nb_comp,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 end

end

function GradPenaltygen(alas::ALASMPCCmod.ALASMPCC,
                        x::Vector,yg::Vector,yh::Vector,
                        rho::Vector,usg::Vector,ush::Vector,
                        uxl::Vector,uxu::Vector,
                        ucl::Vector,ucu::Vector)

 return alas.mod.algoset.penalty(alas.mod.mp,alas.mod.G,alas.mod.H,
                                 alas.mod.nb_comp,
                                 x,yg,yh,rho,
                                 usg,ush,uxl,uxu,ucl,ucu,"grad")
end

function HessPenaltygen(alas::ALASMPCCmod.ALASMPCC,
                        x::Vector,yg::Vector,yh::Vector,
                        rho::Vector,usg::Vector,ush::Vector,
                        uxl::Vector,uxu::Vector,
                        ucl::Vector,ucu::Vector)

 return alas.mod.algoset.penalty(alas.mod.mp,alas.mod.G,alas.mod.H,
                                 alas.mod.nb_comp,
                                 x,yg,yh,rho,
                                 usg,ush,uxl,uxu,ucl,ucu,"hess")
end

function CreatePenaltyNLP(alas::ALASMPCCmod.ALASMPCC,
                          xj::Vector,rho::Vector,
                          usg::Vector,ush::Vector,
                          uxl::Vector,uxu::Vector,
                          ucl::Vector,ucu::Vector)
 n=length(alas.mod.mp.meta.x0)

 penf(x,yg,yh)=Penaltygen(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 penf(x)=penf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 gpenf(x,yg,yh)=GradPenaltygen(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 gpenf(x)=gpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 Hpenf(x,yg,yh)=HessPenaltygen(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 Hpenf(x)=Hpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 pen_mpcc = NLPModels.SimpleNLPModel(x->penf(x), xj,
                                lvar=[alas.mod.mp.meta.lvar;alas.tb*ones(2*alas.mod.nb_comp)], 
                                uvar=[alas.mod.mp.meta.uvar;Inf*ones(2*alas.mod.nb_comp)],
				g=x->gpenf(x),H=x->Hpenf(x))
 return pen_mpcc
end

function UpdatePenaltyNLP(alas::ALASMPCCmod.ALASMPCC,
                          xj::Vector,rho::Vector,
                          usg::Vector,ush::Vector,
                          uxl::Vector,uxu::Vector,
                          ucl::Vector,ucu::Vector,
                          pen_mpcc::NLPModels.AbstractNLPModel)
 n=length(alas.mod.mp.meta.x0)

 penf(x,yg,yh)=Penaltygen(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 penf(x)=penf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])
 gpenf(x,yg,yh)=GradPenaltygen(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 gpenf(x)=gpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])
 Hpenf(x,yg,yh)=HessPenaltygen(alas,x,yg,yh,rho,usg,ush,uxl,uxu,ucl,ucu)
 Hpenf(x)=Hpenf(x[1:n],x[n+1:n+alas.mod.nb_comp],x[n+alas.mod.nb_comp+1:n+2*alas.mod.nb_comp])

 pen_mpcc.f=x->penf(x)
 pen_mpcc.g=x->gpenf(x)
 pen_mpcc.H=x->Hpenf(x)

 return pen_mpcc
end

"""
Mise à jour de rho:
structure de rho :
2*mod.nb_comp
length(mod.mp.meta.lvar)
length(mod.mp.meta.uvar)
length(mod.mp.meta.lcon)
length(mod.mp.meta.ucon)
"""
function RhoUpdate(alas::ALASMPCC,rho::Vector,err::Vector)
 #rho[find(x->x>0,err)]*=alas.rho_update
 #rho*=alas.mod.paramset.rho_update
 rho=min(rho*alas.mod.paramset.rho_update,ones(length(rho))*alas.mod.paramset.rho_max)
 return rho
end

"""
Renvoie : rho_eq, rho_ineq_var, rho_ineq_cons
"""
function RhoDetail(alas::ALASMPCC,rho::Vector)
 nb_ineq_lvar=length(alas.mod.mp.meta.lvar)
 nb_ineq_uvar=length(alas.mod.mp.meta.uvar)
 nb_ineq_lcons=length(alas.mod.mp.meta.lcon)
 nb_ineq_ucons=length(alas.mod.mp.meta.ucon)
 nb_comp=alas.mod.nb_comp

 return rho[1:nb_comp],rho[nb_comp+1:2*nb_comp],rho[2*nb_comp+1:2*nb_comp+nb_ineq_lvar],
        rho[2*nb_comp+nb_ineq_lvar+1:2*nb_comp+nb_ineq_lvar+nb_ineq_uvar],
        rho[2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+1:2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+nb_ineq_lcons],
        rho[2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+nb_ineq_lcons+1:2*nb_comp+nb_ineq_lvar+nb_ineq_uvar+nb_ineq_lcons+nb_ineq_ucons]
end

#end of module
end
