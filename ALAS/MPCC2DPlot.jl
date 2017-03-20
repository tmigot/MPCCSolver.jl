module MPCC2DPlot

using Relaxation
using PyPlot

"""
Plot les contours de la relaxation puis les itérées pour une contrainte de complémentarité
z : tableau de dimension 2
"""
function RelaxationPlot(z::Any,r::Float64,s::Float64,t::Float64,a::Float64,b::Float64)

 #Step 1 : plot des contraintes de relaxations
 x = linspace(-r,b,1000); y = Relaxation.psi(x,r,s,t)
 PyPlot.plot(x, y, color="red", linewidth=2.0, linestyle="-")
 PyPlot.plot(y, x, color="blue", linewidth=2.0, linestyle="-")

 #Step 2 : plot des contraintes de positivité
 x = linspace(-r,b,1000); y = -r*ones(length(x))
 PyPlot.plot(x, y, color="green", linewidth=2.0, linestyle="-")
 PyPlot.plot(y, x, color="green", linewidth=2.0, linestyle="-")

 #Step 3: tracer les itérés
 PyPlot.plot(z[:,1], z[:,2],marker="o",markerfacecolor="black", linestyle="")
 return true
end

#end of module
end
