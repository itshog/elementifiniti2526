# Author: Ivan Bioli (https://github.com/IvanBioli)
# Inspired by code written by Jochen Hinz (https://github.com/JochenHinz) for MATH-451 @ EPFL

using Memoize
using SparseArrays

"""
    initialize_assembly!(mesh::Mesh)

Initialize the assembly process for the given mesh by computing the necessary geometric quantities.

# Arguments
- `mesh::Mesh`: The mesh object for which the assembly is initialized.
"""
function initialize_assembly!(mesh::Mesh)
    get_Bk!(mesh)
    get_detBk!(mesh)
    get_invBk!(mesh)
end

########################### GLOBAL ASSEMBLER ########################### 
"""
    assemble_global(mesh::Mesh, local_assembler!)

Assemble the global stiffness matrix and force vector for the given mesh using the provided local assembler function.

# Arguments
- `mesh::Mesh`: The mesh object.
- `local_assembler!`: A function that assembles the local stiffness matrix and force vector.

# Returns
- `K::SparseMatrixCSC`: The global stiffness matrix.
- `f::Vector`: The global force vector.
"""
function assemble_global(mesh::Mesh, local_assembler!)
    ###########################################################################
    ############################ ADD CODE HERE ################################
    ########################################################################### 
end

########################################################################
########################### LOCAL ASSEMBLERS ###########################
########################################################################

########################### POISSON PROBLEM ###########################
"""
    shapef_2DLFE(quadrule::TriQuad)

Compute the shape functions for the Poisson problem.

# Arguments
- `quadrule::TriQuad`: The quadrature rule.

# Returns
- `shapef`: The shape functions evaluated at the quadrature points.
"""
@memoize function shapef_2DLFE(quadrule::TriQuad)
    q = size(quadrule.points, 2)
    values = zeros(3,q)

    values[1,:] = 1 - quadrule.points[1,:] - quadrule.points[2,:]
    values[2,:] = quadrule.points[1,:]
    values[3,:] = quadrule.points[2,:]

    return values
end

"""
    ∇shapef_2DLFE(quadrule::TriQuad)

Compute the gradients of the shape functions for the Poisson problem.

# Arguments
- `quadrule::TriQuad`: The quadrature rule.

# Returns
- `∇shapef`: The gradients of the shape functions evaluated at the quadrature points.
"""
@memoize function ∇shapef_2DLFE(quadrule::TriQuad)
    q = size(quadrule.points, 2)
    A = [-1 1 0; -1 0 1]
    values = repeat(A, outer = [1,1,q])

    return values
end

"""
    poisson_assemble_local!(Ke::Matrix, fe::Vector, mesh::Mesh, cell_index::Integer, f)

Assemble the local stiffness matrix and force vector for the Poisson problem.

# Arguments
- `Ke::Matrix`: The local stiffness matrix to be assembled.
- `fe::Vector`: The local force vector to be assembled.
- `mesh::Mesh`: The mesh object.
- `cell_index::Integer`: The index of the current cell.
- `f`: The source term function.

# Returns
- `Ke`: The assembled local stiffness matrix.
- `fe`: The assembled local force vector.
"""
function poisson_assemble_local!(Ke::Matrix, fe::Vector, mesh::Mesh, cell_index::Integer, f)
    # Inizializzo a zero stiffness matrix e load vector
    Ke = zeros(3,3)
    fe = zeros(3)

    # Seleziono formula di quadratura
    quadrule = Q2_ref

    # Numero punti di quadratura
    q = size(quadrule.points,2)

    # Dati della trasformazione affine per l'elemento di interesse della mesh
    ak = mesh.ak[:,cell_index]
    Bk = mesh.Bk[:,:,cell_index]
    invBk = mesh.invBk[:,:,cell_index]

    # Definisco i punti di quadratura sull'elemento reale applicando la trasformazione affine (pushforward) ai punti di quadratura sull'elemento di riferimento
    pe = Bk * quadrule.points .+ ak # Valori sui punti, xp = Bk * hat xp + ak

    # I valori delle funzioni sui punti di quadratura reali coincidono con i valori sui punti di quadratura dell'elemento di riferimento
    phie = shapef_2DLFE(quadrule)

    grad_phi = ∇shapef_2DLFE(quadrule)
    grad_phie = invBk[cell_index] * grad_phi
    # grad_phie = mult(invBk,grad_phi,axis=1)

    # grad_phie = mult(invBk, ∇shapef_2DLFE(quadrule, axis=1)
    
    for p=1:q
        wp = quadrule.weights[p] * abs(mesh.detBk[cell_index]) # wp = hat wp * |det Bk|
        for i=1:3
            v = phie[i,p]
            grad_v = grad_phie[:,i,p]
            fe[i] += wp * v * f(pe[:,p]) # Questo e Ke uniche cose che cambiano se implemento FEM per un problema diffusione trasporto o di altro tipo...
            for j=1:3
                grad_u = grad_phie[:,j,p]
                Ke[j,i] += wp * grad_v * grad_u # Questo e fe uniche cose che cambiano se implemento FEM per un problema diffusione trasporto o di altro tipo...
            end
        end
    end

    return Ke, fe
end
