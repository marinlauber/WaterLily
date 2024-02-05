using StaticArrays
using ForwardDiff
using LinearAlgebra: tr, norm, I # this might be an issue
using WaterLily: kern, ∂, inside_u

"""surface integral of pressure"""
function ∮nds_ϵ(p::AbstractArray{T,N},df::AbstractArray{T},body::AutoBody,t=0,ε=1.) where {T,N}
    @WaterLily.loop df[I,:] = p[I]*nds_ϵ(body,loc(0,I,T),t,ε) over I ∈ inside(p)
    [sum(@inbounds(df[inside(p),i])) for i ∈ 1:N] |> Array
end
"""curvature corrected kernel evaluated ε away from the body"""
@inline function nds_ϵ(body::AbstractBody,x,t,ε)
    d,n,_ = measure(body,x,t); κ = 0.5tr(ForwardDiff.hessian(y -> body.sdf(y,t), x))
    κ = isnan(κ) ? 0. : κ;
    n*WaterLily.kern(clamp(d-ε,-1,1))/prod(1.0.+κ*d)
end
# stress tensor
∇²u(J::CartesianIndex{2},u) = @SMatrix [(1+I[i,j])*∂(i,j,J,u) for i ∈ 1:2, j ∈ 1:2]
∇²u(J::CartesianIndex{3},u) = @SMatrix [(1+I[i,j])*∂(i,j,J,u) for i ∈ 1:3, j ∈ 1:3]
"""surface integral of the stress tensor"""
function ∮∇²u_nds(u::AbstractArray{T},df::AbstractArray{T},body::AutoBody,t=0) where {T}
    Nu,n = WaterLily.size_u(u); Inside = CartesianIndices(map(i->(2:i-1),Nu)) #df .= 0.0 
    @WaterLily.loop df[I,:] = ∇²u(I,u)*nds_ϵ(body,loc(0,I,T),t,1.0) over I ∈ Inside
    [sum(@inbounds(df[Inside,i])) for i ∈ 1:n] |> Array
end