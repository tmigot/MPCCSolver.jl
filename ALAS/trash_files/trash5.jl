"""
Line Search with maximum stepsize:

Input :
ma : ActifMPCC
xj : vecteur initial
d : direction précédente en version étendue
"""

function line_search_solve(ma      :: ActifMPCC,
                           xj      :: Vector,
                           d       :: Vector,
                           step    :: Float64,
                           gradpen :: Vector,
                           hg      :: Float64;
                           scaling :: Bool = false)
 #Initialization
 output = 0
 d = redd(ma,d)
 scale = 1.0
 beta = ma.beta
 n, ncc = ma.n, ma.ncc

 #xj est un vecteur de taille (length(bar_wn) x length(bar_w))
 if length(xj) == (n+2*ncc)

  xj=vcat(xj[ma.wnc], xj[n+ma.w13c], xj[n+ncc+ma.w24c])

 elseif length(xj) != (length(ma.wnc) + length(ma.w13c) + length(ma.w24c))

  println("Error dimension : UnconstrainedSolve")
  return
 end

 #Choix de la direction :

 gradf  = grad(ma,xj,gradpen)
 gradft = Array{Float64,1}

 #Calcul d'une direction de descente de taille (n + length(bar_w))

 d = ma.direction(ma, gradf, xj, d, beta)

 slope = dot(gradf, d)

 if slope > 0.0  # restart with negative gradient
  d = - gradf
  slope =  dot(gradf,d)
 end

 #Calcul du pas maximum (peut être infinie!)
 stepmax, wmax, wnew = pas_max(ma, xj, d)

 step,good_grad,ht,nbarmijo,nbwolfe,gradft=ma.linesearch(ma,xj,d,hg,stepmax,scale*slope,step)

 step *= scale

 ols = OutputLS(stepmax, step, slope, beta, nbarmijo, nbwolfe)

 xjp = xj + step*d

 sol  = evalx(ma, xjp)
 dsol = evald(ma, d)

 gradpen = NLPModels.grad(ma.pen.nlp, sol)
 good_grad || (gradft = grad(ma, xjp, gradpen))
 
 s = xjp - xj
 y = gradft - gradf

 #MAJ des paramètres du calcul de la direction
 ma = ma.direction(ma, xjp, beta, gradft, gradf, y, d, step)

 #MAJ du scaling
 if scaling
  scale = dot(y,s)/dot(y,y)
 end
 if scale <= 0.0
  scale=1.0
 end

 small_step = norm(xjp - xj, Inf) <= eps(Float64) ? true : false #on fait un pas trop petit

 #si alpha=pas maximum alors on met w à jour.
 if stepmax == step
  setw(ma, wmax)
 else
  wnew = zeros(Bool,0,0) #si on a pas pris le stepmax alors aucune contrainte n'est ajouté
 end
 if nbarmijo >= ma.paramset.ite_max_armijo || true in isnan.(sol) || true in isnan(ht)
  output = 1
 elseif nbwolfe == ma.paramset.ite_max_wolfe
  output = 2
  xjp = xj
 end

 return sol, ma.w, dsol, step, wnew, output, small_step, ols, gradpen, ht 
end
