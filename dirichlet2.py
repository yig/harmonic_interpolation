'''
Author: Yotam Gingold <yotam (strudel) yotamgingold.com>
License: Public Domain [CC0](http://creativecommons.org/publicdomain/zero/1.0/)

Let the energy E = | G*x - U |^2_M
where:
    x is a row-by-col image flattened into a vector with row-by-col entries.
    G is the gradient operator,
    U are the target gradients (possibly U = G*y for some known function y),
    and M is the diagonal mass matrix which is the norm of each derivative in the summation.
Then E = ( G*x - U ).T * M * ( G*x - U ) = x.T*G.T*M*G*x + U.T*M*U - 2*x.T*G.T*M*U
Let L = G.T*M*G.
Note that G has a row for each row and column derivative in the image.
Let S be a matrix that zeros gradients resulting from the application of G (e.g. G*y).
Formally, let S be an identity matrix whose dimensions are the same as G has rows, modified to
have entries corresponding to the derivatives involving some grid positions to be zero.
Note that S = S*S and S.T = S and S*M = M*S.
Let U = S*G*y for some known other image y, and S has zeros for values in y we don't trust.
Then
E = x.T*L*x + y.T*G.T*S.T*M*S*G*y - 2*x.T*G.T*M*S*G*y
  = x.T*L*x + y.T*G.T*S*M*S*G*y - 2*x.T*G.T*S*M*S*G*y
  = x.T*L*x + y.T*L'*y - 2*x.T*L'*y
where L' = G.T*S*M*S*G = G.T*M*S*G = G.T*S*M*G.
The derivative of 1/2 E with respect to x is:
    0.5 * dE/dx = L*x - L'*y
so the minimizing x could be found by solving
    L*x = L'*y

The G, M, and S in the above description are exactly what is returned by grad_and_mass()
in this module.
mask should be True for every grid location we care about (want a solution to).
skip should be True for every grid location we have a known good value for.
'''

from __future__ import print_function, division

from numpy import *
from scipy import sparse


def grad_and_mass( rows, cols, mask = None, skip = None ):
    '''
    Returns a gradient operator matrix G for a grid of dimensions 'rows' by 'cols',
    a corresponding mass matrix M such that L = G.T*M*G is a
    Laplacian operator matrix that is symmetric and normalized such that the diagonal
    values of interior vertices equal 1,
    and a skip matrix S such that the gradient entries of S*G*x are zero for
    (i,j) such that skip[i,j] is False. If skip is None, all entries are assumed
    to be True and S will be the identity matrix.

    Optional parameter `mask` will result in a gradient operator that entirely
    ignores (i,j) such that mask[i,j] is False.

    In other words, `mask` should be True for every grid location
    you care about (want a solution to via L = G.T*M*G).
    `skip` should be True for every grid location you have a known good value for.

    Matrices returned are scipy.sparse matrices.
    '''

    assert rows > 0 and cols > 0

    if mask is not None:
        mask = asarray( mask, dtype = bool )
        assert mask.shape == ( rows, cols )

    if skip is not None:
        skip = asarray( skip, dtype = bool )
        assert skip.shape == ( rows, cols )

    ## The number of derivatives in the +row direction is cols * ( rows - 1 ),
    ## because the bottom row doesn't have them.
    num_Grow = cols * ( rows - 1 )
    ## The number of derivatives in the +col direction is rows * ( cols - 1 ),
    ## because the right-most column doesn't have them.
    num_Gcol = rows * ( cols - 1 )

    ## coo matrix entries for G
    ijs = []
    vals = []
    ## The diagonal mass matrix.
    mass = []
    ## The diagonal skip matrix.
    S_diag = []

    def ij2index( i,j ):
        assert i >= 0 and i < rows and j >= 0 and j < cols
        return i*cols + j

    output_row = 0
    ## First make the derivatives in the +row direction.
    for i in xrange( rows - 1 ):
        for j in xrange( cols ):
            ## Skip rows involving masked elements.
            if mask is not None and not ( mask[ i, j ] and mask[ i + 1, j ] ): continue
            if skip is not None and not ( skip[ i, j ] and skip[ i + 1, j ] ):
                S_diag.append( 0. )
            else:
                S_diag.append( 1. )

            ijs.append( ( output_row, ij2index( i, j ) ) )
            vals.append( -1 )

            ijs.append( ( output_row, ij2index( i+1, j ) ) )
            vals.append( 1 )

            ## The mass is 0.25 for a non-boundary edge and 0.125 for a boundary edge.
            # mass.append( 0.25 if ( j > 0 and j < cols-1 ) else 0.125 )
            ## UPDATE: It's a little more complicated with a mask, since internal edges
            ##         can be boundary edges.
            m = 0.0
            if j > 0 and ( mask is None or ( mask[ i, j-1 ] and mask[ i + 1, j-1 ] ) ):
                m += 0.125
            if j < cols-1 and ( mask is None or ( mask[ i, j+1 ] and mask[ i + 1, j+1 ] ) ):
                m += 0.125
            mass.append( m )

            output_row += 1
    ## Next make the derivatives in the +col direction.
    for i in xrange( rows ):
        for j in xrange( cols - 1 ):
            ## Skip rows involving masked elements.
            if mask is not None and not ( mask[ i, j ] and mask[ i, j + 1 ] ): continue
            if skip is not None and not ( skip[ i, j ] and skip[ i, j + 1 ] ):
                S_diag.append( 0. )
            else:
                S_diag.append( 1. )

            ijs.append( ( output_row, ij2index( i, j ) ) )
            vals.append( -1 )

            ijs.append( ( output_row, ij2index( i, j+1 ) ) )
            vals.append( 1 )

            ## The mass is 1/4 for a non-boundary edge and 1/8 for a boundary edge.
            # mass.append( 0.25 if ( i > 0 and i < rows-1 ) else 0.125 )
            ## UPDATE: It's a little more complicated with a mask, since internal edges
            ##         can be boundary edges.
            m = 0.0
            if i > 0 and ( mask is None or ( mask[ i-1, j ] and mask[ i-1, j + 1 ] ) ):
                m += 0.125
            if i < rows-1 and ( mask is None or ( mask[ i+1, j ] and mask[ i+1, j + 1 ] ) ):
                m += 0.125
            mass.append( m )

            output_row += 1

    assert len( ijs ) == len( vals )
    assert len( mass ) == output_row

    ## Faster
    '''
    ## Allocate space for a coo matrix entries for G. There are two non-zeros per row of G.
    ijs = empty( ( ( num_Grow + num_Gcol )*2, 2 ), dtype = int )
    vals = empty( ( num_Grow + num_Gcol )*2, dtype = float )

    ## Also allocate space for the diagonal mass matrix.
    mass = empty( num_Grow + num_Gcol, dtype = float )

    ## First make the derivatives in the +row direction.
    ## The pattern of -1 and 1 is very regular. The diagonal is -1 and the 1 is offset from the diagonal by cols.
    vals[ : num_Grow ] = -1
    vals[ num_Grow : 2*num_Grow ] = 1
    ijs[ : num_Grow, : ] = arange( num_Grow )
    ijs[ num_Grow : 2*num_Grow, 0 ] = arange( num_Grow )
    ijs[ num_Grow : 2*num_Grow, 1 ] = arange( num_Grow ) + cols
    mass[ ... ] = ?
    '''

    ## Assert all indices in the G matrix are within the range I expect them to be.
    assert all([ 0 <= i < output_row and 0 <= j < rows*cols for i,j in ijs ])

    G = sparse.coo_matrix( ( vals, asarray(ijs).T ), shape = ( output_row, rows*cols ) )
    assert G.shape == ( output_row, rows*cols )

    # M = sparse.coo_matrix( ( mass, (range(output_row), range(output_row)) ) )
    M = coo_diag( mass )
    assert M.shape == ( output_row, output_row )

    S = coo_diag( S_diag )
    assert S.shape == ( output_row, output_row )

    return G, M, S


def gen_symmetric_grid_laplacian( rows, cols, mask = None ):
    '''
    Returns a Laplacian operator matrix for a grid of dimensions 'rows' by
    'cols'. The matrix is symmetric and normalized such that the diagonal
    values of interior vertices equal 1.

    Matrices returned are scipy.sparse matrices.
    '''

    assert rows > 0 and cols > 0

    G, M, S = grad_and_mass( rows, cols, mask )
    return G.T * M * G


def coo_diag( vals ):
    indices = arange(len( vals ))
    return sparse.coo_matrix( ( vals, (indices, indices) ) )


def test_against_dirichlet():
    print( '=== test_against_dirichlet() ===' )

    import dirichlet
    for rows, cols in [ ( 2, 2 ), ( 2, 3 ), ( 3, 2 ), ( 30, 2 ), ( 2, 30 ), ( 3, 3 ), ( 30, 30 ) ]:
        print( rows, 'rows by', cols, 'cols:', abs( dirichlet.gen_symmetric_grid_laplacian2( rows, cols ) - gen_symmetric_grid_laplacian( rows, cols ) ).sum() )


def test_mask():
    print( '=== test_mask() ===' )

    shape = ( 5,5 )

    mask = ones( shape, dtype = bool )
    mask[2,2] = False

    G, M, S = grad_and_mass( shape[0], shape[1], mask = mask )
    L = G.T*M*G

    # set_printoptions( linewidth = 200 )
    # print( 8*L.todense() )
    ## Print the weight of each grid vertex.
    ## The weight should be the valence (number of neighbors)/4.
    ## Multiply by 4 to get whole numbers
    print( 4*L.diagonal().reshape( shape ) )

if __name__ == '__main__':
    test_against_dirichlet()
    test_mask()
