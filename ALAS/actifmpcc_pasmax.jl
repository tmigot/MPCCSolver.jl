"""
Calcul le pas maximum que l'on peut prendre dans une direction d
d : direction réduit
xj : itéré réduit

output :
alpha : le pas maximum
w_save : l'ensemble des contraintes qui vont devenir actives si choisit alphamax
"""
function PasMax(ma::ActifMPCC,x::Vector,d::Vector)

 if ma.nb_comp>0
  #on récupère les infos sur la contrainte de complémentarité
  alpha_comp,w_save_comp,w_new_comp=PasMaxComp(ma,x,d)
 else
  alpha_comp=Inf
  w_save_comp=zeros(Bool,0,0)
  w_new_comp=ma.w[ma.n+1:ma.n+2*ma.nb_comp,1:2]
 end
 
 alpha_x,w_save_x,w_new_x=PasMaxBound(ma,x,d)
 
 w_save = [w_save_x;w_save_comp]
 w_new  = [w_new_x;w_new_comp]

 alpha  = min(alpha_comp,alpha_x)

 if alpha<0.0
  println("PasMax error: pas maximum négatif.")
  return
 end

 return alpha,w_save,w_new
end

"""
Calcul le pas maximum que l'on peut prendre dans une direction d (par rapport à la contrainte de complémentarité relaxé)
d : direction réduit
xj : itéré réduit

output :
alpha : le pas maximum
w_save : l'ensemble des contraintes qui vont devenir actives si choisit alphamax
"""
function PasMaxComp(ma::ActifMPCC,x::Vector,d::Vector)

 r,s,t = ma.pen.r,ma.pen.s,ma.pen.t

 #initialisation
 alpha=Inf #pas maximum que l'on peut prendre
 w_save=copy(ma.w[ma.n+1:ma.n+2*ma.nb_comp,1:2]) #double tableau des indices avec les contraintes activent en x+alpha*d

 nc = length(ma.wnc)

 #les indices où la première composante est libre
 for i in ma.w13c
  wr13=findfirst(x->x==i,ma.w13c) #l'indice relatif dans les variables libre
  iw13c=ma.n+wr13
  rw13c=nc+wr13

  bloque=(i in ma.w2) && (i in ma.w4)
  if !(i in ma.w24c) && !bloque && d[rw13c]<0
   #on prend le plus petit entre x+alpha*dx>=-r et s+tTheta(x+alpha*dx-s)>=-r
   alpha11=(ma.pen.nlp.meta.lvar[iw13c]-x[rw13c])/d[rw13c]
   alpha12=(Relaxation.invpsi(ma.pen.nlp.meta.lvar[iw13c],r,s,t)-x[rw13c])/d[rw13c]

   alphag=AlphaChoix(alpha,alpha11,alpha12)
   if alphag<=alpha
    alpha=alphag
    w_save=copy(ma.w[ma.n+1:ma.n+2*ma.nb_comp,1:2])
   end

   #update of the active set
   if alpha11==alpha
    w_save[i,1]=true
   end
   if alpha12==alpha
    w_save[i+ma.nb_comp,2]=true
    w_save[i,2]=true
   end

  elseif bloque
   alpha=0.0
  end
 end

 #c'est un copie coller d'au dessus => exporter dans une fonction
 #les indices où la deuxième composante est libre
 for i in ma.w24c
  wr24=findfirst(x->x==i,ma.w24c) #l'indice relatif dans les variables libre
  iw24c=ma.n+length(ma.w13c)+wr24
  rw24c=nc+length(ma.w13c)+wr24

  bloque=(i in ma.w1) && (i in ma.w3)
  if !(i in ma.w13c) && !bloque && d[rw24c]<0
   #on prend le plus petit entre y+alpha*dy>=-r et s+tTheta(y+alpha*dy-s)>=-r
   alpha21=(ma.pen.nlp.meta.lvar[iw24c]-x[rw24c])/d[rw24c]
   alpha22=(Relaxation.invpsi(ma.pen.nlp.meta.lvar[iw24c],r,s,t)-x[rw24c])/d[rw24c]

   alphah=AlphaChoix(alpha,alpha21,alpha22)
   if alphah<=alpha
    alpha=alphah
    w_save=copy(ma.w[ma.n+1:ma.n+2*ma.nb_comp,1:2])
   end

   #on met à jour les contraintes
   if alpha21==alpha
    w_save[i,2]=true
   end
   if alpha22==alpha
    w_save[i+ma.nb_comp,1]=true
    w_save[i,1]=true
   end
  elseif bloque
   alpha=0.0
  end
 end

 #enfin les indices où les deux sont libres
 for i in ma.wc
  #yG-psi(yH)=0 ou yH-psi(yG)=0
  wr1=findfirst(x->x==i,ma.w13c) #l'indice relatif dans les variables libre
  wr2=findfirst(x->x==i,ma.w24c) #l'indice relatif dans les variables libre

  iwr1=wr1+ma.n;iwr2=wr2+length(ma.w13c)+ma.n;
  rwr1=wr1+nc;rwr2=wr2+length(ma.w13c)+nc;

  #alphac=Relaxation.AlphaThetaMax(x[i+ma.n],d[i+ma.n],x[i+length(ma.w13c)+ma.n],d[i+length(ma.w13c)+ma.n],ma.r,ma.s,ma.t)
  alphac=Relaxation.AlphaThetaMax(x[rwr1],d[rwr1],x[rwr2],d[rwr2],r,s,t)
  #yG-tb=0
  #alphac11=d[i+ma.n]<0 ? (ma.pen.nlp.meta.lvar[ma.n+i]-x[i+ma.n])/d[i+ma.n] : Inf
  alphac11=d[rwr1]<0 ? (ma.pen.nlp.meta.lvar[rwr1]-x[rwr1])/d[rwr1] : Inf
  #yH-tb=0
  #alphac21=d[i+length(ma.w13c)+ma.n]<0 ? (ma.pen.nlp.meta.lvar[ma.n+length(ma.w13c)+i]-x[i+length(ma.w13c)+ma.n])/d[i+length(ma.w13c)+ma.n] : Inf  
  alphac21=d[rwr2]<0 ? (ma.pen.nlp.meta.lvar[iwr2]-x[rwr2])/d[rwr2] : Inf  

  alphagh=AlphaChoix(alpha,alphac[1],alphac[2],alphac11,alphac21)

  if alphagh<=alpha
   alpha=alphagh
   w_save=copy(ma.w[ma.n+1:ma.n+2*ma.nb_comp,1:2])
  end

   if alphac[1]==alpha
    w_save[i+ma.nb_comp,1]=true
   end
   if alphac[2]==alpha
    w_save[i+ma.nb_comp,2]=true
   end
   if alphac11==alpha
    w_save[i,1]=true
   end
   if alphac21==alpha
    w_save[i,2]=true
   end

 end #fin boucle for ma.wc

 return alpha,w_save,Array(w_save .& .!ma.w[ma.n+1:ma.n+2*ma.nb_comp,1:2])
end

"""
Met à jour alpha si :
1) il est plus petit
2) il est non-nul
"""
function AlphaChoix(alpha::Float64,alpha1::Float64,alpha2::Float64)
 return AlphaChoix(alpha,alpha1,alpha2,0.0,0.0)
end

function AlphaChoix(alpha::Float64,alpha1::Float64,alpha2::Float64,alpha3::Float64,alpha4::Float64)
 prec=eps(Float64)
 a=alpha1,alpha2,alpha3,alpha4
 a=a[find(x->x>=prec,collect(a))]
 if isempty(a)
  a=max(alpha1,alpha2,alpha3,alpha4)
 end

 return min(minimum(a),alpha)
end

"""
Calcul le pas maximum que l'on peut prendre dans une direction d par rapport aux contraintes de bornes sur x.
d : direction réduit
xj : itéré réduit

output :
alpha : le pas maximum
w_save : l'ensemble des contraintes qui vont devenir actives si choisit alphamax
"""
function PasMaxBound(ma::ActifMPCC,x::Vector,d::Vector)

 alpha = Inf
 w_save=copy(ma.w[1:ma.n,1:2])

 l = ma.pen.nlp.meta.lvar[1:ma.n]
 u = ma.pen.nlp.meta.uvar[1:ma.n]

 nc = length(ma.wnc)
 dp0 = find(x->x>0,d[1:nc])
 dm0 = find(x->x<0,d[1:nc])

 for i=1:nc
  if d[i]>0
   alpha = min(alpha,(u[ma.wnc[i]]-x[i])/d[i])
  elseif d[i]<0
   alpha = min(alpha,(l[ma.wnc[i]]-x[i])/d[i])
  end
 end

 w_save[ma.wnc,1] = (l[ma.wnc]-x[1:nc])./d[1:nc] == 0.0
 w_save[ma.wnc,2] = (u[ma.wnc]-x[1:nc])./d[1:nc] == 0.0

 w_new = Array(w_save .& .!ma.w[1:ma.n,1:2])

 return alpha, w_save,w_new
end
