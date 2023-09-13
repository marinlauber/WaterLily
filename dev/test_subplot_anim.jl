using Plots

n = 100
A = rand(n,100,100)
B = rand(n,100,100)

l = @layout [ a b ; c d]
p1 = surface(1:100,1:100,A[1,:,:],clims=(0,1),legend=false,xticks =false,yticks= false)
p2 = heatmap(A[1,:,:],clims=(0,1),aspect_ratio=1,legend=false,xticks = false,yticks= false)
p3 = heatmap(B[1,:,:],aspect_ratio=1,xticks = false,yticks= false)
p4 = plot(1:100,B[1,1,:],legend=false,xticks=false,yticks= false)
p = plot(p1,p2,p3,p4,layout = l)

anim = @animate for i=1:n
    p[1][1][:z] = A[i,:,:]
    p[2][1][:z] = A[i,:,:]
    p[3][1][:z] = B[i,:,:]
    p[4][1][:y] = B[i,i,:]
end
gif(anim,"example2.gif")