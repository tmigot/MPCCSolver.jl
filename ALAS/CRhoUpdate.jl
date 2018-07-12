"""
Package de fonctions de *********


liste des fonctions :
MinProd(feas::Float64,rho::Vector)

"""
module CRhoUpdate

function MinProd(feas::Float64,rho::Vector)
 return min(feas*maximum(rho),100)
end


#end of module
end
