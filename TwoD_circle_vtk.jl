using WaterLily
using WriteVTK
using Printf: @sprintf

# module vtkWriter

# default writer attributes
_velocity(a::Simulation) = a.flow.u
_pressure(a::Simulation) = a.flow.p
# links a data to a waz of getting the data
default_attrib() = Dict("Velocity"=>_velocity, "Pressure"=>_pressure)
struct vtkWriter
    fname::String
    collection::WriteVTK.CollectionFile
    output_attrib::Dict{String,Function}
    count::Vector{Int}
    function vtkWriter(fname;attrib=default_attrib(),T=Float32)
        new(fname,paraview_collection(fname),attrib,[0])
    end
end
function write!(w::vtkWriter, sim::Simulation)
    k = w.count[1]; N=size(sim.flow.p)
    vtk = vtk_grid(@sprintf("%s_%02i", w.fname, k), [1:n for n in N]...)
    for (name,func) in w.output_attrib
        vtk[name] = size(func(sim))==N ? func(sim) : permutedims(func(sim))
    end
    vtk_save(vtk); w.count[1]=k+1
    w.collection[round(sim_time(sim),digits=4)]=vtk
end
Base.close(w::vtkWriter)=(vtk_save(w.collection);nothing)
function Base.permutedims(a::Array)
    N=length(size(a)); p=[N,1:N-1...]
    return permutedims(a,p)
end
# end

# parameters
L=2^4
Re=250
U =1

# make a body
radius, center = L/2, 3L
Body = AutoBody((x,t)->√sum(abs2, [x[1],x[2],0.0] .- [center,center,0.0]) - radius)
# 2D
sim = Simulation((8L,6L),(U,0),L;U,ν=U*L/Re,body=Body,T=Float64)
# 3D
# sim = Simulation((8L,6L,16),(U,0,0),L;U,ν=U*L/Re,body=Body,T=Float64)

# make a writer with some attributes
velocity(a::Simulation) = a.flow.u
pressure(a::Simulation) = a.flow.p
body(a::Simulation) = a.flow.μ₀
vorticity(a::Simulation) = (@inside a.flow.σ[I] = 
                            WaterLily.curl(3,I,a.flow.u)*a.L/a.U;
                            a.flow.σ)
custom_attrib = Dict(
    "Velocity" => velocity,
    "Pressure" => pressure,
    "Body" => body,
    "Vorticity_Z" => vorticity,
)# this maps what to write to the name in the file
wr = vtkWriter("WaterLily"; attrib=custom_attrib)

# intialize
t₀ = sim_time(sim)
duration = 10
tstep = 0.1

# step and write
@time for tᵢ in range(t₀,t₀+duration;step=tstep)
    # update until time tᵢ in the background
    sim_step!(sim,tᵢ,remeasure=false)

    # write data
    write!(wr, sim)

    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end
close(wr)
