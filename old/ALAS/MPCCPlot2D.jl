export IterationsPlot2D, RelaxationPlot2D

using Relaxation
using PyPlot

function IterationsPlot2D(un::Int64,deux::Int64,
                          trois::Int64,quatre::Int64,
                          orb::OutputRelaxationmod.OutputRelaxation;
                          i::Int64=-1,title::String="",
                          xlabel::String="x_1",
                          ylabel::String="x_2")

 if i<0 #on fait toutes les iterations
  i=1;niter=size(orb.inner_output_alas[i].xtab,1)
 else
  it=i;niter=1;
 end

 for j=i:niter
  figure(j) #on pourrait aussi faire un subplot(111)
  PyPlot.plot(orb.inner_output_alas[j].xtab[un,:], orb.inner_output_alas[j].xtab[deux,:], color="blue",marker="*",label="xk")

  PyPlot.plot(orb.inner_output_alas[j].xtab[trois,:], orb.inner_output_alas[j].xtab[quatre,:], color="green",marker="*",label="sk")

  ub=max(maximum(orb.inner_output_alas[j].xtab[un,:]),maximum(orb.inner_output_alas[j].xtab[deux,:]))
  RelaxationPlot2D(ub,orb.rtab[j],j,orb)

  if xlabel!=""
   PyPlot.xlabel(xlabel)
  end
  if xlabel!=""
   PyPlot.ylabel(ylabel)
  end
  if title!=""
   PyPlot.title(title)
  end
 end

 nothing
end

function RelaxationPlot2D(ub::Float64,
                          tb::Float64,
                          i::Int64,
                          orb::OutputRelaxationmod.OutputRelaxation;
                          legend::String="Butterfly",
                          pas::Float64=0.1)


 x1=orb.stab[i]:pas:ub
 x2=collect(Relaxation.psi(x1,orb.rtab[i],orb.stab[i],orb.ttab[i]))

 PyPlot.plot(x1, x2, color="red",label=legend)
 PyPlot.plot(x2, x1, color="red")
 PyPlot.plot(-tb:pas:ub,-tb*ones(-tb:pas:ub), color="red")
 PyPlot.plot(-tb*ones(-tb:pas:ub),-tb:pas:ub, color="red")
 PyPlot.plot(0:pas:ub,zeros(0:pas:ub), color="black")
 PyPlot.plot(zeros(0:pas:ub),0:pas:ub, color="black")
 PyPlot.axis("equal")
 PyPlot.legend(loc="upper left")

 nothing
end
