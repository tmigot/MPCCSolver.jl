function setw(ma :: ActifMPCC,
              w  :: Array{Bool,2})

 ma.wnew = w .& .!ma.w
 ma.w = w

 return updatew(ma)
end

function updatew(ma :: ActifMPCC)

 ncc = ma.ncc
 n       = ma.n

 #le vecteur x d'avant (ne dépend pas de ma.w)
 x = evalx(ma, ma.x0)

 #on actualise avec w
 ma.wnc = find(.!ma.w[1:n,1] .& .!ma.w[1:n,2])
 ma.wn1 = find(ma.w[1:n,1])
 ma.wn2 = find(ma.w[1:n,2])

 ma.w1  = find(ma.w[n+1:n+ncc,1])
 ma.w2  = find(ma.w[n+1:n+ncc,2])
 ma.w3  = find(ma.w[n+ncc+1:n+2*ncc,1])
 ma.w4  = find(ma.w[n+ncc+1:n+2*ncc,2])

 ma.wcomp = find(ma.w[n+ncc+1:n+2*ncc,1] .| ma.w[n+ncc+1:n+2*ncc,2])

 ma.w13c  = find(.!ma.w[n+1:n+ncc,1] .& .!ma.w[n+ncc+1:n+2*ncc,1])
 ma.w24c  = find(.!ma.w[n+1:n+ncc,2] .& .!ma.w[n+ncc+1:n+2*ncc,2])

 ma.wc = find(.!ma.w[n+1:n+ncc,1] .& .!ma.w[n+1:n+ncc,2] .& .!ma.w[n+ncc+1:n+2*ncc,1] .& .!ma.w[n+ncc+1:n+2*ncc,2])

 ma.wcc = find((ma.w[n+1:n+ncc,1] .| ma.w[n+ncc+1:n+2*ncc,1]) .& (ma.w[n+1:n+ncc,2] .| ma.w[n+ncc+1:n+2*ncc,2]))

 ma.x0 = vcat(x[ma.wnc], x[ma.n+ma.w13c], x[ma.n+ma.ncc+ma.w24c])

 return ma
end

function setbeta(ma :: ActifMPCC,
                 b  :: Float64)
 ma.beta = b
 return ma
end

function setcrho(ma   :: ActifMPCC,
                 crho :: Float64)
 ma.crho = crho
 return ma
end

function sethess(ma   :: ActifMPCC,
                 Hess :: Array{Float64,2})
 ma.Hess = Hess
 return ma
end
