module OutputLSmod

type OutputLS
 stepmax  :: Float64
 step     :: Float64
 slope    :: Float64
 beta     :: Float64
 nbarmijo :: Float64
 nbwolfe  :: Float64
end

#Initialisation
#function OutputLS()
# return OutputLS()
#end

#Mise à jour
#function Update(ols::OutputLS)
# return ols
#end

#Print
function Print(ols     :: OutputLS,
               n       :: Int64,
               verbose :: Int64)
 if verbose>2
  println(" stepmax=",ols.stepmax," step=",ols.step," slope=",ols.slope," beta=",ols.beta," nbarmijo=",ols.nbarmijo," nbwolfe=",ols.nbwolfe)
 end

 return true
end

#end of module
end
