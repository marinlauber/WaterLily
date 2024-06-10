# structure to store fluid state
struct Store
    uˢ::AbstractArray
    pˢ::AbstractArray
    xˢ::AbstractArray
    ẋˢ::AbstractArray
    function Store(sim::AbstractSimulation)
        xs = [copy(b.surf.pnts) for b in sim.body.bodies]
        vs = [copy(b.velocity.pnts) for b in sim.body.bodies]
        new(copy(sim.flow.u),copy(sim.flow.p),xs,vs)
    end
end
function store!(s::Store,sim::AbstractSimulation)
    s.uˢ .= sim.flow.u; s.pˢ .= sim.flow.p;
    for i ∈ 1:length(sim.body.bodies)
        s.xˢ[i] .= sim.body.bodies[i].surf.pnts
        s.ẋˢ[i] .= sim.body.bodies[i].velocity.pnts
    end
end
function revert!(s::Store,sim::AbstractSimulation)
    sim.flow.u .= s.uˢ; sim.flow.p .= s.pˢ; pop!(sim.flow.Δt)
    for i ∈ 1:length(sim.body.bodies)
        sim.body.bodies[i].surf.pnts     .= s.xˢ[i]
        sim.body.bodies[i].velocity.pnts .= s.ẋˢ[i]
    end
end
# unpack subarray of inceasing values of an hcat
function unpack(a)
    tmp=[a[1]]; ks=Vector{Number}[]
    for i ∈ 2:length(a)
        if a[i]>=a[i-1]
            push!(tmp,a[i])
        else
            push!(ks,tmp); tmp=[a[i]]
        end
    end
    push!(ks,tmp)
    return ks
end
function knotVectorUnpack(knots)
    knots = reshape(knots,reverse(size(knots)))[1,:]
    unpack(knots)
end
function getControlPoints(points, knots)
    points = reshape(points,reverse(size(points)))
    ncp = [length(knot)-count(i->i==0,knot) for knot in knots]
    @assert sum(ncp) == size(points,2) "Number of control points does not match the number of points"
    return [points[:,1+sum(ncp[1:i])-ncp[1]:sum(ncp[1:i])] for i in 1:length(ncp)]
end
function quadPointUnpack(quadPoints)
    quadPoints = reshape(quadPoints,reverse(size(quadPoints)))
    quadPoints = [filter(!isone,filter(!iszero,quadPoints[:,i]))[1] for i in 1:size(quadPoints,2)]
    unpack(quadPoints)
end
function getDeformation(points,knots)
    ncp = [length(knot)-count(i->i==0,knot) for knot in knots]
    return [points[:,1+sum(ncp[1:i])-ncp[1]:sum(ncp[1:i])] for i in 1:length(ncp)]
end

using PreCICE
function initialize!(;KnotMesh="KnotMesh",ControlPointMesh="ControlPointMesh",ForceMesh="ForceMesh")
    # initilise PreCICE
    PreCICE.initialize()

    # get the mesh verticies from the fluid solver
    (_, knots) = getMeshVertexIDsAndCoordinates(KnotMesh)
    knots = knotVectorUnpack(knots)
   
    # get the mesh verticies from the structural solver
    (ControlPointsID, ControlPoints) = getMeshVertexIDsAndCoordinates(ControlPointMesh)
    ControlPointsID = Array{Int32}(ControlPointsID)
    ControlPoints = getControlPoints(ControlPoints, knots)
    
    (quadPointID, quadPoint) = getMeshVertexIDsAndCoordinates(ForceMesh)
    forces = zeros(reverse(size(quadPoint))...)
    quadPointID = Array{Int32}(quadPointID)
    quadPoint = quadPointUnpack(quadPoint)
    return ControlPointsID, ControlPoints, quadPointID, quadPoint, forces, knots
end