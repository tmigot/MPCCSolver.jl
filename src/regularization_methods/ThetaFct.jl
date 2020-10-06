###############################################################################
#Package of functions related to the Theta functions.
#
#Citation:
#Haddou, M., & Maheux, P. (2014).
#Smoothing methods for nonlinear complementarity problems.
#Journal of Optimization Theory and Applications, 160(3), 711-729.
###############################################################################

"""
Definition of Theta^1

function theta(x :: Union{Float64,Vector},
               r :: Float64)
output : Union{Float64,Vector}
"""
function theta(x :: Union{Float64,Vector},
               r :: Float64)

 if r < 0.0
  throw("DomainError: Theta undefined for r < 0.")
 elseif r==0.0 && in(0.0,x)
  throw("DomainError: Theta undefined for x=0, r=0.")
 end

 y = (x .>= 0) .* x ./ (abs.(x) .+ r) + (x .< 0) .* (x ./ r-x.^2 ./ r^2)

 return y
end

"""
Inverse of function theta
function invtheta(x :: Union{Float64,Vector},
                  r :: Float64)
"""
function invtheta(x :: Union{Float64,Vector},
                  r :: Float64)

 if in(true, x .> 1.)
  throw("DomainError: Theta^{-1} undefined for x >= 1.")
 end

 y = (x .>= 0) .* -r .* x ./ (x .- 1) + (x .< 0) .* r/2 .* (1 .- sqrt.(1 .+ 4 .* abs.(x)))

 return y
end

"""
First derivative of theta
function dtheta(x :: Union{Float64,Vector},
                r :: Float64)
"""
function dtheta(x :: Union{Float64,Vector},
                r :: Float64)
 if r < 0.0
  throw("DomainError: Theta undefined for r < 0.")
 elseif r==0.0 && in(0.0,x)
  throw("DomainError: Theta undefined for x = 0, r = 0.")
 end

 y = (x .>= 0) .* (r ./ (abs.(x) .+ r).^2) + (x .< 0) .* (1/r .-2 .* x ./ r^2)

 return y
end

"""
Second derivative of theta
function ddtheta(x :: Union{Float64,Vector},
                 r :: Float64)
"""
function ddtheta(x :: Union{Float64,Vector},
                 r :: Float64)

 if r < 0.0
  throw("DomainError: Theta undefined for r < 0.")
 elseif r==0.0 && in(0.0,x)
  throw("DomainError: Theta undefined for x = 0, r = 0.")
 end

 y = (x .>= 0) .* (-2*r ./ (abs.(x) .+ r).^3) + (x .< 0) .* (-2/(r^2))

 return y
end
#Fun fact:
# ThetaFct.ddtheta([-1.0], 0.1) != ThetaFct.ddtheta(-1.0, 0.1)

"""
alpha_theta_max : Solve the equation in alpha where psi(x,r,s,t)=s+t*theta(x,r)
In : x  (position of x)
     dx (direction of x)
     y  (position of y)
     dy (direcion of y)
     r,s,t (parameters)

output: Array with two entries
"""
function alpha_theta_max(x  :: Float64,
                         dx :: Float64,
                         y  :: Float64,
                         dy :: Float64,
                         r  :: Float64,
                         s  :: Float64,
                         t  :: Float64)

 alphaf1 = _alpha_theta_max_one_leaf(x, dx, y, dy, r, s, t)
 alphaf2 = _alpha_theta_max_one_leaf(y, dy, x, dx, r, s, t)

 #le premier fixe yg et le deuxième yh
 return [alphaf1, alphaf2]
end
function alpha_theta_max(x  :: Vector,
                         dx :: Vector,
                         y  :: Vector,
                         dy :: Vector,
                         r  :: Float64,
                         s  :: Float64,
                         t  :: Float64)

 alphaf1 = _alpha_theta_max_one_leaf(x, dx, y, dy, r, s, t)
 alphaf2 = _alpha_theta_max_one_leaf(y, dy, x, dx, r, s, t)

 #le premier fixe yg et le deuxième yh
 alpha_yg = minimum(alphaf1[alphaf1 .>= 0.0])
 alpha_yh = minimum(alphaf2[alphaf2 .>= 0.0])
 return [alpha_yg, alpha_yh]
end

"""
function _alpha_theta_max_one_leaf(x  :: Union{Float64,Vector},
                                   dx :: Union{Float64,Vector},
                                   y  :: Union{Float64,Vector},
                                   dy :: Union{Float64,Vector},
                                   r  :: Float64,
                                   s  :: Float64,
                                   t  :: Float64)

Solve the equation (a variable) :

(x+a*dx-s)*(y+a*dy-s+r)-t*(y+a*dy-s)=0 -> alphaf1p

comments: (solve the equation)
r^2[x+a*dx-s-t*((y+a*dy-s)/r-(y+a*dy-s)^2/r^2)]=0 -> alphaf1n
"""
function _alpha_theta_max_one_leaf(x  :: Union{Float64,Vector},
                                   dx :: Union{Float64,Vector},
                                   y  :: Union{Float64,Vector},
                                   dy :: Union{Float64,Vector},
                                   r  :: Float64,
                                   s  :: Float64,
                                   t  :: Float64)

 if t!=0

  a = dx .* dy
  b = dy .* (x .- s .- t) .+ dx .* (y .- s .+ r)
  c = x .* y - x .* s + x .* r - y .* s .+ s^2 .- s*r-t .* y .+ t*s

  alphaf1p = _poly_degree_2_positif(a, b, c)

  tmp1 = (alphaf1p .>= 0) .& ((x+alphaf1p .* dx .>= s) .& (y+alphaf1p .* dy .>= s))

  #si on considère la contrainte intérieure:
  #a = dy.^2
  #b = r^2.*dx-r*t.*dy+2*t*y.*dy-2*t*s.*dy
  #c = r^2.*x .- r^2*s - r*t*y .+ r*t*s+t.*y.^2.+t*s^2-t*2*s.*y
  #alphaf1n = _poly_degree_2_positif(a, b, c)
  #tmp2 = (alphaf1p .>= 0) .& (x+alphaf1p.*dx .<= s .& y+alphaf1p.*dy .<= s)

  alphaf1 = (tmp1) .* alphaf1p + (!tmp1) .* Inf

 else #cas où t=0

  tmp = (dx .!= 0) .& (((s .- x) ./ dx .> 0) .| ((s .== x) .& (dx .> 0)))
  alphaf1 = (tmp) .* (s .- x)./dx + (!tmp) .* Inf

 end #fin de condition t!=0

 return alphaf1
end

"""
function _poly_degree_2_positif(a :: Union{Float64,Vector},
                                b :: Union{Float64,Vector},
                                c :: Union{Float64,Vector})

Compute the smallest positive root of a second order polynomial
or a negative/complex value if no positive real root.
If no solution or infinite nb of solutions, return Inf.
"""
function _poly_degree_2_positif(a :: Union{Float64,Vector},
                                b :: Union{Float64,Vector},
                                c :: Union{Float64,Vector})

 disc = b.^2 .- 4 .* a .* c

 if in(true, disc.<0.0)
  return -Inf.*a #throw("FunctionError: complex roots.")
 end

 sol1 = (-b .+ sqrt.(disc)) ./ (2 .* a)
 sol2 = (-b .- sqrt.(disc)) ./ (2 .* a)

 solp  = ((sol1 .>= 0.0) .& (sol2 .>= 0.0)) .* min.(sol1,sol2)
 #solp += xor.(sol1 .< 0.0, sol2 .< 0.0) .* max.(sol1,sol2)
 solp += ((sol1 .< 0.0) .| (sol2 .< 0.0)) .* max.(sol1,sol2)

 sol  = (a .!= 0.) .* solp
 sol += ((a .== 0.) .& (b .!= 0.)) .* (-c ./ b)
 sol += ((a .== 0.) .& (b .== 0.)) .* Inf

 return sol
end
