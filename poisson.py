from __future__ import print_function, division

from numpy import *
import scipy.sparse.linalg

def normals_to_target_gradients( normals ):
    pass

def gradient_operator( rows, cols, mask = None ):
    '''
    Given dimensions of a `rows` by `cols` 2D grid,
    and optional parameter `mask`, a rows-by-cols boolean array, where False values are ignored in the optimization,
    returns the gradient operator G as a sparse matrix such that
    G * the grid of data reshaped into a vector produces the vector containing
    the gradient of the grid in the i and j directions as a
    flattened rows-by-cols-by-2 array.
    '''
    
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
    and `linear_constraints`, a sequence of ( [ row, col, coeffs ], rhs ) representing linear equation constraints,
    returns a matrix C and vector Crhs such that solving the system of equations C*x = Crhs
    produces a vector (a flattened rows-by-cols 2D array) of values x that satisfy
    the linear constraints in the least squares sense.
    '''
    
    assert len( linear_constraints ) > 0
    
    count = 0
    
    ijs = []
    vals = []
    ## #constraints-by-k
    b = zeros( ( len( linear_constraints ), len( linear_constraints[0][1] ) ) )
    
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

def solve_poisson_simple( rows, cols, target_gradients, linear_constraints = None, linear_constraints_weight = None, mask = None ):
    '''
    Given dimensions of a `rows` by `cols` 2D grid,
    a rows-by-cols-by-2-by-k array of target gradient values at each element of the grid `target_gradients` in the i and j directions,
    and optional parameters:
        `linear_constraints`: a sequence of ( [ row, col, coeff ], rhs ) representing linear equation constraints,
        `linear_constraints_weight`: a single weight value for all of the constraints,
        `mask`: a rows-by-cols boolean array, where False values are ignored in the optimization
    solves the specified poisson equation to obtain an array of rows-by-cols-by-k values
    whose gradient matches the target gradients subject to the optional linear constraints
    and mask as closely as possible.
    '''
    
    print( 'solve_poisson_simple( rows = %s, cols = %s, target_gradients.shape = %s, |constraints| = %s, linear_constraints_weight = %s, |mask == False| = %s )' % (
        rows, cols,
        target_gradients.shape,
        len( linear_constraints ),
        linear_constraints_weight,
        None if mask is None else ( mask == False ).sum()
        ) )
    
    assert ( linear_constraints is None ) == ( linear_constraints_weight is None )
    
    G = gradient_operator( rows, cols, mask = mask )
    L = G.T*G
    
    system = L
    rhs = G.T * target_gradients.reshape( rows*cols*2, -1 )
    
    if linear_constraints is not None and len( linear_constraints ) > 0:
        C, Crhs = generate_constraint_matrix( rows, cols, linear_constraints )
        
        system += linear_constraints_weight * C
        rhs += linear_constraints_weight * Crhs
    
    x = scipy.sparse.linalg.spsolve( system, rhs )
    
    return x.reshape( rows, cols, -1 )

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
        ## Put a high value in the upper-left corner if we're testing cut edges
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
    
    sol = solve_poisson_simple( rows, cols, zeros( ( rows, cols, 2 ) ), linear_constraints = linear_constraints, linear_constraints_weight = 1e5, mask = mask )
    
    from PIL import Image
    from recovery import normalize_to_char_img
    Image.fromarray( normalize_to_char_img( sol ) ).save( 'sol_poisson.png' )
    print( '[Saved "sol_poisson.png".]' )
    #heightmesh.save_grid_as_OBJ( sol_nomask, 'sol_hard.obj' )
    
    from pprint import pprint
    print( sol )

if __name__ == '__main__':
    test_poisson_simple()
