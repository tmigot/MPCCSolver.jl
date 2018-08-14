function relaxation_rule!(ma   :: ActifMPCC,
                          xj   :: Vector,
                          l    :: Vector,
                          wmax :: Array{Bool,2})

  copy_wmax = copy(wmax)

  n   = ma.n
  ncc = ma.ncc

  llx  = l[1:n]
  lux  = l[n+1:2*n]
  lg   = l[2*n+1:2*n+ncc]
  lh   = l[2*n+ncc+1:2*n+2*ncc]
  lphi = l[2*n+2*ncc+1:2*n+3*ncc]

  # Relaxation de l'ensemble d'activation : 
  # désactive toutes les contraintes négatives
  ll = [llx; lg; lphi; lux; lh; lphi] #pas très catholique comme technique
  ma.w[find(x -> x<0, ll)] = zeros(Bool, length(find(x -> x<0, ll)))

  # Règle d'anti-cyclage : 
  # on enlève pas une contrainte qui vient d'être ajouté.
  ma.w[find(x->x==1.0, copy_wmax)] = ones(Bool,length(find(x->x==1.0, copy_wmax)))

 return updatew(ma)
end
