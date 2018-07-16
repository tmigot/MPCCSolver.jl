function SlackComplementarityProjection(alas::ALASMPCC)

 nb_comp=alas.mod.nb_comp

 if nb_comp==0
  return alas.xj
 end

 #initialisation :
 #x=[alas.mod.xj;alas.mod.G(alas.mod.xj);alas.mod.H(alas.mod.xj)]
 x=[alas.xj;NLPModels.cons(alas.mod.G,alas.xj);NLPModels.cons(alas.mod.H,alas.xj)]
 n=length(x)-2*nb_comp

 #projection sur les contraintes de positivité relaxés : yG>=tb et yH>=tb
 x[n+1:n+nb_comp]=max.(x[n+1:n+nb_comp],ones(nb_comp)*alas.tb)
 x[n+nb_comp+1:n+2*nb_comp]=max.(x[n+nb_comp+1:n+2*nb_comp],ones(nb_comp)*alas.tb)

 #projection sur les contraintes papillons : yG<=psi(yH,r,s,t) et yH<=psi(yG,r,s,t)
 for i=1:nb_comp
  psiyg=psi(x[n+i],alas.r,alas.s,alas.t)
  psiyh=psi(x[n+nb_comp+i],alas.r,alas.s,alas.t)

  if x[n+i]-psiyh>0 && x[n+nb_comp+i]-psiyg>0
   x[n+i]>=x[n+nb_comp+i] ? x[n+nb_comp+i]=psiyg : x[n+i]=psiyh
  end
 end

 return x
end
