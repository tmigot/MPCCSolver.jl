module StoppingMPCCmod

using MPCCmod
using OutputRelaxationmod

using NLPModels

type StoppingMPCC

 #paramètres pour la résolution du MPCC
 #precmpcc::Float64
 #paramin::Float64 #valeur minimal pour les paramères (r,s,t)

 #variables de sorties
 optimal::Bool
 realisable::Bool
 solved::Bool
 param::Bool

end

function start(mpccsol,xk,r,s,t)

 real=viol_contrainte_norm(mpccsol.mod,xk)
 realisable=real<=mpccsol.paramset.precmpcc
 solved=true
 param=(t+r+s)>mpccsol.paramset.paramin
 f=MPCCmod.obj(mpccsol.mod,xk)
 or=OutputRelaxationmod.OutputRelaxation(xk,real, f)
 #heuristic in case the initial point is the solution
 optimal=realisable && stationary_check(mpccsol.mod,xk,mpccsol.paramset.precmpcc)
 OK=param && !(realisable && solved && optimal)

 sts=StoppingMPCC(optimal,realisable,solved,param)

 return sts,OK, or
end

function stop(mpccsol,xk,r,s,t,output,solved,or,sts)

  real=viol_contrainte_norm(mpccsol.mod,xk[1:mpccsol.mod.n])
  f=MPCCmod.obj(mpccsol.mod,xk[1:mpccsol.mod.n])

  sts.solved=true in isnan.(xk)?false:solved
  sts.realisable=real<=mpccsol.paramset.precmpcc

 if sts.solved && sts.realisable
  sts.optimal=!isnan(f) && !(true in isnan.(xk)) && stationary_check(mpccsol.mod,xk[1:mpccsol.mod.n],mpccsol.paramset.precmpcc)
 end

 sts.param=(t+r+s)>mpccsol.paramset.paramin

 OK=sts.param && !sts.optimal
 or=OutputRelaxationmod.UpdateOR(or,xk[1:mpccsol.mod.n],0,r,s,t,mpccsol.paramset.prec_oracle(r,s,t,mpccsol.paramset.precmpcc),real,output,f)
 
 return OK
end


"""
Donne la norme 2 de la violation des contraintes avec slack

note : devrait appeler viol_contrainte
"""
function viol_contrainte_norm(mod::MPCCmod.MPCC,x::Vector,yg::Vector,yh::Vector;tnorm::Real=2)
 return norm(MPCCmod.viol_contrainte(mod,x,yg,yh),tnorm)
end

function viol_contrainte_norm(mod::MPCCmod.MPCC,x::Vector;tnorm::Real=2) #x de taille n+2nb_comp

 n=mod.n
 if length(x)==n
  resul=max(viol_comp_norm(mod,x),viol_cons_norm(mod,x))
 else
  resul=MPCCmod.viol_contrainte_norm(mod,x[1:n],x[n+1:n+mod.nb_comp],x[n+mod.nb_comp+1:n+2*mod.nb_comp],tnorm=tnorm)
 end
 return resul
end

"""
Donne la norme de la violation de la complémentarité min(G,H)
"""
function viol_comp_norm(mod::MPCCmod.MPCC,x::Vector;tnorm::Real=2)
 return mod.nb_comp>0?norm(MPCCmod.viol_comp(mod,x),tnorm):0
end

"""
Donne la norme de la violation des contraintes \"classiques\"
"""
function viol_cons_norm(mod::MPCCmod.MPCC,x::Vector;tnorm::Real=2)

 n=mod.n
 x=length(x)==n?x:x[1:n]
 feas=norm([max.(mod.mp.meta.lvar-x,0);max.(x-mod.mp.meta.uvar,0)],tnorm)

 if mod.mp.meta.ncon !=0

  c=NLPModels.cons(mod.mp,x)
  feas=max(norm([max.(mod.mp.meta.lcon-c,0);max.(c-mod.mp.meta.ucon,0)],tnorm),feas)

 end

 return feas
end

"""
Donne la violation de la réalisabilité dual (norme Infinie)
"""
function dual_feasibility_norm(mod::MPCCmod.MPCC,x::Vector,l::Vector,A::Any,precmpcc::Float64) #type général pour matrice ?
 return optimal=norm(MPCCmod.dual_feasibility(mod,x,l,A),Inf)<=precmpcc
end
"""
Vérifie les signes de la M-stationarité (l entier)
"""
function sign_stationarity_check(mod::MPCCmod.MPCC,x::Vector,l::Vector,precmpcc::Float64)

  Il=find(z->norm(z-mod.mp.meta.lvar,Inf)<=precmpcc,x)
  Iu=find(z->norm(z-mod.mp.meta.uvar,Inf)<=precmpcc,x)

  IG=[];IH=[];Ig=[];Ih=[];

 if mod.mp.meta.ncon+mod.nb_comp >0

  c=cons(mod.mp,x)
  Ig=find(z->norm(z-mod.mp.meta.lcon,Inf)<=precmpcc,c)
  Ih=find(z->norm(z-mod.mp.meta.ucon,Inf)<=precmpcc,c)

  if mod.nb_comp>0
   IG=find(z->norm(z-mod.G.meta.lcon,Inf)<=precmpcc,NLPModels.cons(mod.G,x))
   IH=find(z->norm(z-mod.H.meta.lcon,Inf)<=precmpcc,NLPModels.cons(mod.H,x))
  end
 end

 #setdiff(∪(Il,Iu),∩(Il,Iu))
 l_pos=max.(l[1:2*n+2*mod.mp.meta.ncon],0)
 I_biactif=∩(IG,IH)
 lG=[2*n+2*mod.mp.meta.ncon+I_biactif]
 lH=[2*n+2*mod.mp.meta.ncon+mod.nb_comp+I_biactif]
 l_cc=min.(lG.*lH,max.(-lG,0)+max.(-lH,0))

 return norm([l_pos;l_cc],Inf)<=precmpcc
end

"""
Vérifie les signes de la M-stationarité (l actif)
"""
function sign_stationarity_check(mod::MPCCmod.MPCC,x::Vector,l::Vector,
                                 Il::Array{Int64,1},Iu::Array{Int64,1},
                                 Ig::Array{Int64,1},Ih::Array{Int64,1},
                                 IG::Array{Int64,1},IH::Array{Int64,1},precmpcc::Float64)

 nl=length(Il)+length(Iu)+length(Ig)+length(Ih)
 nccG=length(IG)
 nccH=length(IH)
 l_pos=max.(l[1:nl],0)
 I_biactif=∩(IG,IH)
 lG=l[I_biactif+nl]
 lH=l[nl+nccG+I_biactif]
 l_cc=min.(lG.*lH,max.(-lG,0)+max.(-lH,0))

 return norm([l_pos;l_cc],Inf)<=precmpcc
end

"""
For a given x, compute the multiplier and check the feasibility dual
"""
function stationary_check(mod::MPCCmod.MPCC,x::Vector,precmpcc::Float64)
 n=mod.n
 b=-MPCCmod.grad(mod,x)

 if mod.mp.meta.ncon+mod.nb_comp ==0

  optimal=norm(b,Inf)<=precmpcc

 else
  A, Il,Iu,Ig,Ih,IG,IH=MPCCmod.jac_actif(mod,x,precmpcc)

  if !(true in isnan.(A) || true in isnan.(b))
   l=pinv(full(A))*b #pinv not defined for sparse matrix
   optimal=dual_feasibility_norm(mod,x,l,A,precmpcc)
   good_sign=sign_stationarity_check(mod,x,l,Il,Iu,Ig,Ih,IG,IH,precmpcc)
  else
   @printf("Evaluation error: NaN in the derivative")
   optimal=false
  end
 end

 return optimal && good_sign
end


#end of module
end
