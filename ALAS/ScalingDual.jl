"""
Package de fonctions de scaling sur la réalisabilité duale
output: un nombre \sigma

\|grad Lag(x,\lambda) \|_\infty  <= epsilon_{dual}*\sigma


liste des fonctions :
NoScaling(usg::Vector,ush::Vector,
                   uxl::Vector,uxu::Vector,
                   ucl::Vector,ucu::Vector,
                   lg::Vector,lh::Vector,lphi::Vector,
                   precrst::Float64,prec::Float64,
                   rho::Vector,dualfeas::Float64)
ParamScaling(usg::Vector,ush::Vector,
                   uxl::Vector,uxu::Vector,
                   ucl::Vector,ucu::Vector,
                   lg::Vector,lh::Vector,lphi::Vector,
                   precrst::Float64,prec::Float64,
                   rho::Vector,dualfeas::Float64)

"""
module ScalingDual

function NoScaling(usg::Vector,ush::Vector,
                   uxl::Vector,uxu::Vector,
                   ucl::Vector,ucu::Vector,
                   lg::Vector,lh::Vector,lphi::Vector,
                   precrst::Float64,prec::Float64,
                   rho::Vector,dualfeas::Float64)
 return 1
end

function ParamScaling(usg::Vector,ush::Vector,
                   uxl::Vector,uxu::Vector,
                   ucl::Vector,ucu::Vector,
                   lg::Vector,lh::Vector,lphi::Vector,
                   precrst::Float64,prec::Float64,
                   rho::Vector,dualfeas::Float64)

 temp=norm([usg;ush;uxl;uxu;ucl;ucu;lg;lh;lphi])

 return max(temp,1)
end

#end of module
end
