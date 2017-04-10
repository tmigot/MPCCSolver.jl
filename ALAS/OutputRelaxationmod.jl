module OutputRelaxationmod

using OutputALASmod

type OutputRelaxation

 solved::Int64 #renvoi 0 si réussi
 solve_message::AbstractString #message de sortie
 xtab::Array #tableau des itérés
 inner_output_alas::Any #en fait une liste de OutputALAS

 rtab::Array{Float64,1}
 stab::Array{Float64,1}
 ttab::Array{Float64,1}
 prectab::Array{Float64,1}

 realisabilite::Array{Float64,1}
 objtab::Array{Float64,1}

end

#Initialisation:
function OutputRelaxation(x0::Vector,realisabilite::Float64,obj::Float64)
 solved=0
 solve_message="Success"
 xtab=Array{Float64,2}
 xtab=collect(x0)
 inner_output=[]
 rtab=[]
 stab=[]
 ttab=[]
 prectab=[]
 
 return OutputRelaxation(solved,solve_message,xtab,inner_output,rtab,stab,ttab,prectab,[realisabilite],[obj])
end

#Après une itération :
function UpdateOR(or::OutputRelaxation,xk::Vector,stat::Int64,r::Float64,s::Float64,t::Float64,prec::Float64,realisabilite::Float64,outputalas::OutputALASmod.OutputALAS,obj::Float64)
 if stat==0
  or.solved=0
  or.solve_message="Success"
 else #stat !=0
  or.solved=-1
  or.solve_message="Fail"
 end

 or.xtab=[or.xtab xk]
 or.inner_output_alas=[or.inner_output_alas;outputalas]
 or.rtab=[or.rtab;r]
 or.stab=[or.stab;s]
 or.ttab=[or.ttab;t]
 or.prectab=[or.prectab;prec]
 or.realisabilite=[or.realisabilite;realisabilite]
 or.objtab=[or.objtab;obj]
 
 return or
end

function UpdateOR(or::OutputRelaxation,xk::Vector,stat::Int64,r::Float64,s::Float64,t::Float64,prec::Float64,realisabilite::Float64,output,obj::Float64)
 if stat==0
  or.solved=0
  or.solve_message="Success"
 else #stat !=0
  or.solved=-1
  or.solve_message="Fail"
 end

 or.xtab=[or.xtab xk]
 or.inner_output_alas=[or.inner_output_alas;output]
 or.rtab=[or.rtab;r]
 or.stab=[or.stab;s]
 or.ttab=[or.ttab;t]
 or.prectab=[or.prectab;prec]
 or.realisabilite=[or.realisabilite;realisabilite]
 or.objtab=[or.objtab;obj]
 
 return or
end

function Print(or::OutputRelaxation,n::Int64,verbose::Int64)
 if verbose>0
  println("")
  j=1
  println("j :",0," (r,s,t)=(-,-,-) eps=- xj=",or.xtab[1:n,j]," f(xj)=",or.objtab[j]," \|c(xj)\|=",or.realisabilite[j])
  for r in or.rtab
   println("j :",j," (r,s,t)=(",or.rtab[j],",",or.stab[j],",",or.ttab[j],") eps=",or.prectab[j],"\n xj=",or.xtab[1:n,j+1]," f(xj)=",or.objtab[j+1]," \|c(xj)\|=",or.realisabilite[j+1])
   OutputALASmod.Print(or.inner_output_alas[j],n,verbose)
   j+=1
  end
 end

 return true
end

#end of module
end
