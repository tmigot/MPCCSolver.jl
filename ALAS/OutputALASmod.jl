module OutputALASmod

using OutputLSmod

type OutputALAS

 solved::Int64 #renvoi 0 si réussi
 solve_message::AbstractString #message de sortie
 xtab::Array #tableau des itérés
 dtab::Array #tableau des directions
 rhotab::Array{Float64,1}
 steptab::Array{Float64,1}

 realisabilite::Array{Float64,1}
 dualrealisabilite::Array{Float64,1}

 inner_output_linesearch::Any #en fait une liste de OutputLS

end

#Initialisation
function OutputALAS(x0::Vector,dj::Vector,feas::Float64,dualfeas::Float64,rho::Vector)
 return OutputALAS(0,"Success",x0,dj,[norm(rho,Inf)],[],[feas],[dualfeas],[])
end

#Mise à jour
function Update(oa::OutputALAS,xk::Vector,rho::Vector,feas::Float64,dualfeas::Float64,dj::Vector,step::Float64,outputls::OutputLSmod.OutputLS)

 oa.realisabilite=[oa.realisabilite;feas]
 oa.dualrealisabilite=[oa.dualrealisabilite;dualfeas]
 oa.rhotab=[oa.rhotab;norm(rho,Inf)]
 oa.steptab=[oa.steptab;step]
 oa.xtab=[oa.xtab xk]
 oa.dtab=[oa.dtab dj]
 oa.inner_output_linesearch=[oa.inner_output_linesearch;outputls]

 return oa
end

#Print
function Print(oa::OutputALAS,n::Int64,verbose::Int64)
 if verbose>1
  j=1
  nb_comp=Int64(0.5*(length(oa.xtab)/length(oa.realisabilite)-n))
  for step in oa.steptab
   println("k :",j," xjk=",oa.xtab[1:n,j+1]," sg=",oa.xtab[n+1:n+nb_comp,j+1]," sh=",oa.xtab[n+nb_comp+1:n+2*nb_comp,j+1],"\n\|c(xjk)\|=",oa.realisabilite[j+1]," JL(xjk)=",oa.dualrealisabilite[j+1]," rho= ",oa.rhotab[j+1]," \nalpha=",oa.steptab[j]," d=",oa.dtab[1:n+2*nb_comp,j+1])
   OutputLSmod.Print(oa.inner_output_linesearch[j],n,verbose)
   j+=1
  end
  println("")
 end

 return true
end

#end of module
end
