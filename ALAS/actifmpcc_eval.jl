function evalx(ma :: ActifMPCC,
               x  :: Vector)

 r,s,t = ma.pen.r,ma.pen.s,ma.pen.t

 if length(x) != ma.n+2*ma.ncc

  nc = length(ma.wnc)
  nw13c = nc+length(ma.w13c)

  #construction du vecteur de taille n+2ncc que l'on évalue :
  xf = s*ones(ma.n+2*ma.ncc)
  xf[ma.wnc] = x[1:nc]
  xf[ma.w13c+ma.n] = x[nc+1:nw13c]
  xf[ma.w24c+ma.n+ma.ncc] = x[nw13c+1:nw13c+length(ma.w24c)]

  #on regarde les variables x fixées:
  xf[ma.wn1] = ma.pen.nlp.meta.lvar[ma.wn1]
  xf[ma.wn2] = ma.pen.nlp.meta.uvar[ma.wn2]

  #on regarde les variables yG fixées :
  xf[ma.w1+ma.n] = ma.pen.nlp.meta.lvar[ma.w1+ma.n]
  xf[ma.w3+ma.n] = psi(xf[ma.w3+ma.n+ma.ncc], r, s, t)

  #on regarde les variables yH fixées :
  xf[ma.w2+ma.n+ma.ncc] = ma.pen.nlp.meta.lvar[ma.w2+ma.n+ma.ncc]
  xf[ma.w4+ma.n+ma.ncc] = psi(xf[ma.w4+ma.n], r, s, t)

 else

  xf = x

 end

 return xf
end

function redx(ma :: ActifMPCC,
              x  :: Vector)

 nc = length(ma.wnc)
 nw13c = nc+length(ma.w13c)

 xf = zeros(nc+length(ma.w13c)+length(ma.w24c))
 xf[1:nc] = x[ma.wnc]
 xf[nc+1:nw13c] = x[ma.w13c+ma.n]
 xf[nw13c+1:nw13c+length(ma.w24c)] = x[ma.w24c+ma.ncc+ma.n]

 return xf
end

"""
Renvoie la direction d au complet (avec des 0 aux actifs)
"""
function evald(ma :: ActifMPCC,
               d  :: Vector)

 nc = length(ma.wnc)
 nw13c = nc+length(ma.w13c)

 df = zeros(ma.n+2*ma.ncc)
 df[ma.wnc] = d[1:nc]
 df[ma.w13c+ma.n] = d[nc+1:nw13c]
 df[ma.w24c+ma.ncc+ma.n] = d[nw13c+1:nw13c+length(ma.w24c)]

 return df
end

"""
Renvoie la direction d réduite
"""
function redd(ma :: ActifMPCC,
              d  :: Vector)

 nc = length(ma.wnc)
 nw13c = nc+length(ma.w13c)

 df = zeros(nc+length(ma.w13c)+length(ma.w24c))
 df[1:nc] = d[ma.wnc]
 df[nc+1:nw13c] = d[ma.w13c+ma.n]
 df[nw13c+1:nw13c+length(ma.w24c)] = d[ma.w24c+ma.ncc+ma.n]

 return df
end

function redd(ma :: ActifMPCC,
              d  :: Vector,
              w  :: Array{Int64,1})

  df = zeros(length(w))
  df[1:length(w)] = d[w]

 return df
end
