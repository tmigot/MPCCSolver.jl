#AM to MPCC
export AMtoMPCC

using AmplNLReader

function AMtoMPCC(AM::AmplModel)
 
 nc=find(x->x == 0,AM.cvar)
 ncc=find(x->x > 0,AM.cvar)
 n_cc=length(ncc)
 ncon=length(nc)
 nvar=AM.meta.nvar

 #pas optimal comme technique :
 Temp=sparse(zeros(n_cc,nvar))
 for i=1:n_cc
  Temp[i,ncc[i]]=1.0
 end
 maj(t)=Temp'*t 
 Temp2=sparse(zeros(ncon,nvar))
 for i=1:ncon
  Temp2[i,nc[i]]=1.0
 end
 maj2(t)=Temp2'*t


#doublon de la contrainte de positivité...
nlp=SimpleNLPModel(x->obj(AM,x),AMnlp.meta.x0,
                   lvar=AM.meta.lvar,uvar=AM.meta.uvar,
                   c=x->c(x)[nc],lcon=AM.meta.lcon[nc],
                   ucon=AM.meta.ucon[nc],g=x->grad(AM,x),
                   H=(x;y=zeros(ncon),obj_weight=1.0)->hess(AM,x;y=maj2(y),obj_weight=obj_weight),
                   J=x->jac(AM,x)[nc,1:nvar],
                   Jtp=(x,v)->jac(AM,x)[nc,1:nvar]'*v)

G=SimpleNLPModel(()->(),AM.meta.x0,c=x->c(x)[ncc],
                   lcon=AM.meta.lcon[ncc],
                   ucon=AM.meta.ucon[ncc],J=x->jac(AM,x)[ncc,1:nvar],
                   Jtp=(x,v)->jac(AM,x)[ncc,1:nvar]'*v,
                   H=(x;y=zeros(n_cc),obj_weight=1.0)->hess(AM,x;y=maj(y),obj_weight=0.0))

H=SimpleNLPModel(()->(),AM.meta.x0,c=x->x[ncc],
                   lcon=AM.meta.lvar[ncc],
                   ucon=AM.meta.uvar[ncc],
                   J=x->Temp,Jtp=(x,v)->maj(v),
                   H=(x;y=zeros(n_cc),obj_weight=1.0)->zeros(nvar,nvar))

 return nlp,G,H
end
