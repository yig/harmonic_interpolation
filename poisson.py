from __future__ import print_function, division

from numpy import *
import scipy.sparse.linalg

def solve_poisson_simple( target_gradients, linear_constraints = None, linear_constraints_weight = None, mask = None, bilaplacian = False ):
    '''
    Given a rows-by-cols-by-2-by-k array of target gradient values at each k-dimensional element of the grid `target_gradients` in the i and j directions,
    and optional parameters:
        `linear_constraints`: a sequence of ( [ row, col, coeff ], rhs ) representing linear equation constraints,
        `linear_constraints_weight`: a single weight value for all of the constraints,
        `mask`: a rows-by-cols boolean array, where False values are ignored in the optimization
    solves the specified poisson equation to obtain an array of rows-by-cols-by-k values
    whose gradients matches the target gradients subject to the optional linear constraints
    and mask as closely as possible.
    
    NOTE: For a scalar-valued function such as a height field, k=1.
    '''
    
    print( 'solve_poisson_simple( target_gradients.shape = %s, |constraints| = %s, linear_constraints_weight = %s, |mask == False| = %s )' % (
        target_gradients.shape,
        len( linear_constraints ),
        linear_constraints_weight,
        None if mask is None else ( mask == False ).sum()
        ) )
    
    ## Check the shape of the target_gradients array.
    assert len( target_gradients.shape ) == 4
    rows, cols = target_gradients.shape[:2]
    
    ## Our task is impossible if the grid is 1x1.
    assert rows > 0 and cols > 0
    
    ## Gradients on the grid must be two-dimensional.
    assert target_gradients.shape[2] == 2
    
    ## If there is a mask, it must have the same first two dimensions.
    assert mask is None or mask.shape == target_gradients.shape[:2]
    
    ## If linear_constraints is not None, then the weight must be given.
    assert ( linear_constraints is None ) or ( linear_constraints_weight is not None )
    
    ## Actually, linear_constraints are the only way to specify constraints.
    ## The poisson system is under-constrained without any constraints.
    assert len( linear_constraints ) > 0
    
    ## Build the gradient and laplacian operators.
    G = gradient_operator( rows, cols, mask = mask )
    L = G.T*G
    
    ## TODO: A mass matrix.
    # print( L.diagonal() )
    ## One over mass
    # ooMass = scipy.sparse.identity( L.shape[0] )
    # ooMass.setdiag( 1./sqrt(L.diagonal()) )
    # G = G*ooMass
    # L = G.T*G
    # print( L.diagonal() )
    
    if bilaplacian:
        G = G*L
        L = L.T*L
    
    system = L
    rhs = G.T * target_gradients.reshape( rows*cols*2, -1 )
    
    ## Add the linear constraints if present.
    if linear_constraints is not None and len( linear_constraints ) > 0:
        C, Crhs = generate_constraint_matrix( rows, cols, linear_constraints )
        
        system += linear_constraints_weight * C
        rhs += linear_constraints_weight * Crhs
    
    ## Use cvxopt.choldmod if we have it.
    try:
        import cvxopt, cvxopt.cholmod
        system = system.tocoo()
        system = cvxopt.spmatrix( system.data, asarray( system.row, dtype = int ), asarray( system.col, dtype = int ) )
        rhs = cvxopt.matrix( rhs )
        cvxopt.cholmod.linsolve( system, rhs )
        x = array( rhs ).squeeze()
    except ImportError:
        print( 'No cvxopt.cholmod, using scipy.sparse.linalg.spsolve()' )
        x = scipy.sparse.linalg.spsolve( system, rhs ).squeeze()
    
    return x.reshape( rows, cols, -1 )

def normals_to_target_gradients( normals ):
    '''
    Given a rows-by-cols-by-3 array 'normals' containing a 3D normal at each grid location,
    returns `target_gradients` suitable for passing to solve_poisson_simple(),
    a rows-by-cols-by-2-by-1 array of target gradient values at each element
    of the grid in the i and j directions,
    
    NOTE: The 3D coordinate space of the normal is ( x = +i, y = +j, z = direction obtained by cross( +i, +j ) ).
          This is a right-hand coordinate system.
    
    NOTE: To be valid normals on a height map, all normals must have z > 0.
    '''
    
    ## Our task is impossible if the grid is 1x1.
    rows, cols = normals.shape[:2]
    assert rows > 0 and cols > 0
    assert normals.shape == ( rows, cols, 3 )
    
    ## Normals must all be front-facing.
    ## With a height field, we can't actually handle silhouettes.
    #assert ( normals[:,:,2] > 0 ).all()
    
    num_back_facing = ( normals[:,:,2] < -1e-5 ).sum()
    if num_back_facing > 0:
        print( "Found %s back-facing normals. That's impossible for a height field. Replacing them with flat." % ( num_back_facing, ) )
    
    num_silhouette = ( abs( normals[:,:,2] ) <= 1e-5 ).sum()
    if num_silhouette > 0:
        print( "Found %s silhouette normals. That's impossible for a height field. Replacing them with flat." % ( num_silhouette, ) )
    
    if num_back_facing + num_silhouette > 0:
        normals = normals.copy()
        normals[ ( normals[:,:,2] <= 1e-5 )[...,newaxis] ] = ( 0,0,1 )
    
    ## Let z be the scalar function defined on the grid.
    ## We will create gradients dz/dx and dz/dy by pretending that there is
    ## a triangle with vertices (i,j), (i+1,j), (i,j+1) whose normal is given.
    ## Viewing the triangle's (i,j), (i+1,j) edge in the xz plane,
    ## the normal has a slope (rise/run) of normal.z / normal.x.
    ## The dx constraint is the slope of the surface's z value
    ## in the x direction.  This slope should be perpendicular to
    ## normal.z / normal.x,
    ## therefore dx = -normal.x / normal.z.
    ## Similarly for the dy constraint.
    
    target_gradients = zeros( ( rows, cols, 2, 1 ) )
    ## dz/dx
    target_gradients[ :,:,0,0 ] = -normals[:,:,0]/normals[:,:,2]
    ## dz/dy
    target_gradients[ :,:,1,0 ] = -normals[:,:,1]/normals[:,:,2]
    
    return target_gradients

def gradient_operator( rows, cols, mask = None ):
    '''
    Given dimensions of a `rows` by `cols` 2D grid,
    and optional parameter `mask`, a rows-by-cols boolean array, where False values are ignored in the optimization,
    returns the gradient operator G as a sparse matrix such that
    G * the grid of data reshaped into a vector produces the vector containing
    the gradient of the grid in the i and j directions as a
    flattened rows-by-cols-by-2 array.
    '''
    
    ## If the mask is present, it must be rows-by-cols.
    assert mask is None or mask.shape == ( rows, cols )
    
    ijs = []
    vals = []
    
    def ij2index( i,j ):
        '''Vectorizes the input rows-by-cols array'''
        return i*cols + j
    
    def ijk2index( i,j,k ):
        '''Vectorizes the output rows-by-cols-by-2 array'''
        return ( i*cols + j )*2 + k
    
    for i in range( rows-1 ):
        for j in range( cols ):
            if mask is not None and mask[ i,j ] != mask[ i+1,j ]: continue
            
            ijs.append( ( ijk2index( i,j,0 ), ij2index( i,j ) ) )
            vals.append( -1 )
            
            ijs.append( ( ijk2index( i,j,0 ), ij2index( i+1,j ) ) )
            vals.append( 1 )
    
    for i in range( rows ):
        for j in range( cols-1 ):
            if mask is not None and mask[ i,j ] != mask[ i,j+1 ]: continue
            
            ijs.append( ( ijk2index( i,j,1 ), ij2index( i,j ) ) )
            vals.append( -1 )
            
            ijs.append( ( ijk2index( i,j,1 ), ij2index( i,j+1 ) ) )
            vals.append( 1 )
    
    ijs = array( ijs, dtype = int )
    G = scipy.sparse.coo_matrix( ( vals, ijs.T ), shape = ( rows*cols*2, rows*cols ) ).tocsr()
    
    return G

def generate_constraint_matrix( rows, cols, linear_constraints ):
    '''
    Given dimensions of a `rows` by `cols` 2D grid
    and `linear_constraints`, a sequence of ( [ ( row, col, coeff ), ... ], rhs ) representing linear equation constraints,
    returns a matrix C and vector Crhs such that solving the system of equations C*x = Crhs
    produces a vector (a flattened rows-by-cols 2D array) of values x that satisfy
    the linear constraints in the least squares sense.
    '''
    
    assert len( linear_constraints ) > 0
    
    count = 0
    
    ijs = []
    vals = []
    ## #constraints-by-K
    K = len( linear_constraints[0][1] )
    b = zeros( ( len( linear_constraints ), K ) )
    
    def ij2index( i,j ):
        return i*cols + j
    
    for constraint_coeffs, rhs in linear_constraints:
        for row, col, coeff in constraint_coeffs:
            ijs.append( ( count, ij2index( row, col ) ) )
            vals.append( coeff )
        
        b[ count ] = rhs
        
        count += 1
    
    ijs = array( ijs, dtype = int )
    A = scipy.sparse.coo_matrix( ( vals, ijs.T ), shape = ( count, rows*cols ) ).tocsr()
    
    C = A.T*A
    Crhs = A.T*b
    
    return C, Crhs

def normalize_to_char_array( arr ):
    '''
    Given a 2D scalar array, returns a uint8 array by
    linearly mapping the values to lie within the range [0,255].
    '''
    
    ## Ensure the input is floating point.
    arr = asfarray( arr ).squeeze()
    
    ## It must be 2D.
    assert len( arr.shape ) == 2
    
    ## Get the max and min.
    max_val, min_val = arr.max(), arr.min()
    
    ## If it is all the same value, make it zeros.
    if max_val == min_val:
        return zeros( arr.shape, dtype = uint8 )
    
    ## Normalize it.
    arr = ( arr - min_val )/( max_val - min_val )
    
    ## Scale and clip it to [0,255].
    arr = ( 255.*arr ).round().clip( 0, 255 ).astype( uint8 )
    
    return arr

def test_poisson_simple():
    what = 'large'
    if what == 'large':
        rows = 250
        cols = 250
        br0 = 100
        br1 = 150
        bc0 = br0
        bc1 = br1
    elif what == 'small':
        rows = 9
        cols = 9
        br0 = 3
        br1 = 6
        bc0 = br0
        bc1 = br1
    else:
        raise RuntimeError, "what"
    
    test_mask = True
    
    ## This works with K = 1 or K = 3.
    K = 1
    linear_constraints = [
        ## Put a high value in the upper-left corner if we're testing the mask
        ( [ ( 0, 0, 1. ) ], [br0*2. if test_mask else 0.]*K ),
        ( [ ( rows-1, 0, 1. ) ], [0.]*K ),
        ( [ ( 0, cols-1, 1. ) ], [0.]*K ),
        ( [ ( rows-1, cols-1, 1. ) ], [0.]*K ),
        ( [ ( br0, bc0, 1. ) ], [br0]*K ),
        ( [ ( br1-1, bc0, 1. ) ], [br0]*K ),
        ( [ ( br0, bc1-1, 1. ) ], [br0]*K ),
        ( [ ( br1-1, bc1-1, 1. ) ], [br0]*K )
        ]
    
    if test_mask:
        mask = zeros( ( rows, cols ), dtype = bool )
        mask[br0:br1,bc0:bc1] = True
    else:
        mask = None
    
    test_normals = True
    if test_normals:
        normals = zeros( ( rows, cols, 3 ) )
        normals[:,:,2] = 1.
        target_gradients = normals_to_target_gradients( normals )
    else:
        target_gradients = zeros( ( rows, cols, 2, 1 ) )
    
    ## With a bilaplacian, it is no longer the poisson equation.
    test_bilaplacian = False
    
    sol = solve_poisson_simple( target_gradients, linear_constraints = linear_constraints, linear_constraints_weight = 1e5, mask = mask, bilaplacian = test_bilaplacian )
    
    name = 'sol_poisson-%smask%s' % ( '' if test_mask else 'no', '-bilaplacian' if test_bilaplacian else '' )
    
    from PIL import Image
    Image.fromarray( normalize_to_char_array( sol ) ).save( name + '.png' )
    print( '[Saved "%s.png".]' % name )
    # import heightmesh
    # heightmesh.save_grid_as_OBJ( sol, name + '.obj' )
    
    # from pprint import pprint
    # print( sol )

if __name__ == '__main__':
    test_poisson_simple()
