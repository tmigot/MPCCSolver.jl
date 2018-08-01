function SlackComplementarityProjection(alas::ALASMPCC)

 nb_comp=alas.mod.nb_comp
 n=alas.mod.n

 x = copy(alas.xj)

 if nb_comp==0
  return x
 end

 #Initialisation :
 if length(x) == n
  x = vcat(x,consG(x),consH(x))
 else
  x = x
 end

 #projection sur les contraintes de bornes: l <= x <= u
 l = alas.mod.mp.meta.lvar
 u = alas.mod.mp.meta.uvar
 x[1:n] = x[1:n] + max.(l-x[1:n],0) + max.(x[1:n]-u,0)

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
