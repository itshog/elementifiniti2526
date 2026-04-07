# Author: Ivan Bioli (https://github.com/IvanBioli)
using LinearAlgebra

"""
    triarea(V1, V2, V3)

Calculate the area of a triangle given its vertices.

# Arguments
- `V1`: The first vertex of the triangle.
- `V2`: The second vertex of the triangle.
- `V3`: The third vertex of the triangle.

# Returns
- `area::Float64`: The area of the triangle.
"""
function triarea(V1, V2, V3)
    a = norm(V1-V2)
    b = norm(V1-V3)
    c = norm(V2-V3)
    s = 0.5 * (a+b+c)
    area = sqrt(s * (s-a) * (s-b) * (s-c))
    return area
end

"""
    Q0(p, T, u)

Perform numerical integration using the Q0 quadrature rule (i.e., baricenter formula) over a mesh.
This quadrature rule has order 1.

# Arguments
- `p::Matrix`: The coordinates of the mesh nodes.
- `T::Matrix`: The connectivity matrix of the mesh elements.
- `u::Function`: The function to be integrated.

# Returns
- `I_approx::Float64`: The approximate integral of the function over the mesh.
"""
function Q0(p, T, u)
    num_triangles = size(T,2)
    baricenters = zeros(2,num_triangles)
    areas = zeros(num_triangles)
    values = zeros(num_triangles)
    for j=1:num_triangles
        baricenters[:,j] = 1/3 * (p[:,T[1,j]] + p[:,T[2,j]] + p[:,T[3,j]])
        areas[j] = triarea(p[:,T[1,j]], p[:,T[2,j]], p[:,T[3,j]])
        values[j] = u(baricenters[:,j])
    end
    return dot(areas,values)
end

"""
    Q1(p, T, u)

Perform numerical integration using the Q1 quadrature rule (i.e., vertex formula) over a mesh.
This quadrature rule has order 1.

# Arguments
- `p::Matrix`: The coordinates of the mesh nodes.
- `T::Matrix`: The connectivity matrix of the mesh elements.
- `u::Function`: The function to be integrated.

# Returns
- `I_approx::Float64`: The approximate integral of the function over the mesh.
"""
function Q1(p, T, u)
    num_triangles = size(T,2)
    areas = zeros(num_triangles)
    values = zeros(3,num_triangles)
    for j=1:num_triangles
        areas[j] = triarea(p[:,T[1,j]], p[:,T[2,j]], p[:,T[3,j]])
        for i=1:3
            values[i,j] = u(p[:,T[i,j]])
        end
    end
    means = 1/3 * sum(values, dims=1)
    return dot(areas,means)
end

"""
    Q2(p, T, u)

Perform numerical integration using the Q2 quadrature rule (i.e., midpoints rule) over a mesh.
This quadrature rule has order 2.

# Arguments
- `p::Matrix`: The coordinates of the mesh nodes.
- `T::Matrix`: The connectivity matrix of the mesh elements.
- `u::Function`: The function to be integrated.

# Returns
- `I_approx::Float64`: The approximate integral of the function over the mesh.
"""
function Q2(p, T, u)
    num_triangles = size(T,2)
    areas = zeros(num_triangles)
    values = zeros(3,num_triangles)
    m = zeros(2,3,num_triangles)
    for j=1:num_triangles
        areas[j] = triarea(p[:,T[1,j]], p[:,T[2,j]], p[:,T[3,j]])
        m[:,1,j] = 1/2 * (p[:,T[1,j]] + p[:,T[2,j]])
        m[:,2,j] = 1/2 * (p[:,T[2,j]] + p[:,T[3,j]])
        m[:,3,j] = 1/2 * (p[:,T[1,j]] + p[:,T[3,j]])
        for i=1:3
            values[i,j] = u(m[:,i,j])
        end
    end
    means = 1/3 * sum(values, dims=1)
    return dot(areas,means)
end
