#!/usr/bin/env python

from numpy import *
from math import acos
import itertools
from itertools import izip as zip

def recovery1( mesh ):
    ## 1 Take only front-facing triangles of the mesh.
    ## 2 Fix the boundaries.
    ## 3 Attach normals to interior vertices.
    ## TODO Q: Why not attach to faces?
    ## A1: Then I need a laplace operator on faces, and this is strange when my boundaries are edges.
    ## 4 Solve a poisson system for values at vertices.
    
    ## 1
    from trimesh import TriMesh
    output = TriMesh()
    
    vmap = [-1] * len( mesh.vs )
    vcount = 0
    output_face2mesh_face = dict()
    for fi, face in enumerate( mesh.faces ):
        if mesh.face_normals[ fi ][2] < 0.: continue
        
        for v in face:
            if vmap[ v ] == -1:
                vmap[ v ] = vcount
                vcount += 1
                output.vs.append( mesh.vs[v] )
        
        output_face2mesh_face[ len( output.faces ) ] = fi
        output.faces.append( tuple( [ vmap[v] for v in face ] ) )
    
    output.vs = asarray( output.vs )
    
    ## 2
    '''
    boundary_verts = set()
    boundary_vert_normal_sum = {}
    boundary_vert_num_normals = {}
    for (i,j) in output.boundary_edges():
        boundary_verts.add( i )
        boundary_verts.add( j )
        ## The vector from vertex i to vertex j.
        edge_vec = asarray( output.vs[j] ) - asarray( output.vs[i] )
        ## Pretend it's a perfectly aligned silhouette edge, as in our input data.
        edge_vec[2] = 0.
        edge_vec *= 1./sqrt(dot( edge_vec, edge_vec ))
        ## 90 degree clockwise turn
        n = array( [ edge_vec[1], -edge_vec[0], 0. ] )
        boundary_vert_normal_sum[i] = boundary_vert_normal_sum.setdefault( i, zeros( 3 ) ) + n
        boundary_vert_normal_sum[j] = boundary_vert_normal_sum.setdefault( j, zeros( 3 ) ) + n
        boundary_vert_num_normals[i] = boundary_vert_num_normals.setdefault( i, 0 ) + 1
        boundary_vert_num_normals[j] = boundary_vert_num_normals.setdefault( j, 0 ) + 1
    
    assert sorted( boundary_vert_normal_sum.keys() ) == sorted( boundary_vert_num_normals.keys() )
    assert set( boundary_vert_normal_sum.keys() ) == boundary_verts
    assert len( boundary_vert_normal_sum ) == len( boundary_verts )
    '''
    
    
    ## Solve for 'output' in-place and return 'output' as well.
    #output = solve_nonlinear( output, [ ( ofi, mesh.face_normals[ mfi ] ) for ofi, mfi in output_face2mesh_face.iteritems() ] )
    output = solve_linear( output, [ ( ofi, mesh.face_normals[ mfi ] ) for ofi, mfi in output_face2mesh_face.iteritems() ] )
    output.write_OBJ( 'output.obj' )

def zero_dx_dy_constraints_from_cut_edges( cut_edges ):
    '''
    Given a parameter 'cut_edges' suitable for passing to
    'gen_symmetric_grid_laplacian()',
    returns two values ( dx_constraints, dy_constraints ) suitable for
    passing to 'solve_grid_linear()' that enforce derivatives of zero across
    the edges in 'cut_edges'.  This allows 'solve_grid_linear()' to solve
    for a continuous surface with discontinuous derivatives.
    
    tested
    '''
    
    dx_constraints = []
    dy_constraints = []
    for ( i, j ), ( k, l ) in cut_edges:
        assert i == k or j == l
        assert abs( i - k ) + abs( j - l ) == 1
        
        if 0 != i - k:
            dx_constraints.append( ( min( i, k ), j, 0. ) )
        else:
            dy_constraints.append( ( i, min( j, l ), 0. ) )
    
    assert len( dx_constraints ) + len( dy_constraints ) == len( cut_edges )
    return dx_constraints, dy_constraints

def cut_edges_from_mask( mask ):
    '''
    Given a boolean 2D array 'mask', returns 'cut_edges' suitable for passing
    to 'gen_symmetric_grid_laplacian()' where the horizontal and vertical
    transitions between True and False entries becomes cut edges.
    
    tested
    (see also test_cut_edges())
    '''
    
    mask = asarray( mask, dtype = bool )
    
    cut_edges = []
    
    horiz = mask[:-1,:] != mask[1:,:]
    for i,j in zip( *where( horiz ) ):
        cut_edges.append( ( ( i,j ), ( i+1, j ) ) )
    del horiz
    
    vert = mask[:,:-1] != mask[:,1:]
    for i,j in zip( *where( vert ) ):
        cut_edges.append( ( ( i,j ), ( i, j+1 ) ) )
    del vert
    
    return cut_edges

def assert_valid_cut_edges( rows, cols, cut_edges ):
    assert rows > 0 and cols > 0
    
    if len( cut_edges ) > 0:
        ## cut edge indices must both be inside the grid
        def in_grid( i,j ): return i >= 0 and i < rows and j >= 0 and j < cols
        assert all([ in_grid( i,j ) and in_grid( k,l ) for (i,j), (k,l) in cut_edges ])
        ## cut edges must be horizontal or vertical neighbors
        assert all([ abs( i - k ) + abs( j - l ) == 1 for (i,j), (k,l) in cut_edges ])
        ## cut edges must be unique
        assert len( frozenset([ ( tuple( ij ), tuple( kl ) ) for ij, kl in cut_edges ]) ) == len( cut_edges )

def gen_symmetric_grid_laplacian1( rows, cols, cut_edges = None ):
    '''
    Returns a Laplacian operator matrix for a grid of dimensions 'rows' by 'cols'.
    The matrix is symmetric and normalized such that the diagonal values of interior vertices equal 1.
    
    Optional parameter 'cut_edges', a sequence of 2-tuples ( (i,j), (k,l) )
    where (i,j) and (k,l) must be horizontal or vertical neighbors in the grid,
    specifies edges in the grid which should be considered to be disconnected.
    
    tested
    (see also test_cut_edges())
    '''
    
    assert rows > 0
    assert cols > 0
    
    if cut_edges is None: cut_edges = []
    
    assert_valid_cut_edges( rows, cols, cut_edges )
    
    from scipy import sparse
    
    N = rows
    M = cols
    def ind2ij( ind ):
        assert ind >= 0 and ind < N*M
        return ind // M, ind % M
    def ij2ind( i,j ):
        assert i >= 0 and i < N and j >= 0 and j < M
        return i*M + j
    
    Adj = []
    for i in xrange( 0, rows ):
        for j in xrange( 0, cols ):
            ind00 = ij2ind( i,j )
            
            ## If I wanted to remove these conditionals, I could do
            ## one loop for 1...-1 and another for 0 and another for -1.
            ## I could also only add adjacencies in positive coordinate
            ## directions and then do "AdjM += AdjM.T" to get symmetry (the negative direction).
            if i+1 < rows:
                indp0 = ij2ind( i+1,j )
                Adj.append( ( ind00, indp0 ) )
            
            if j+1 < cols:
                ind0p = ij2ind( i,j+1 )
                Adj.append( ( ind00, ind0p ) )
            
            if i > 0:
                indm0 = ij2ind( i-1,j )
                Adj.append( ( ind00, indm0 ) )
            
            if j > 0:
                ind0m = ij2ind( i,j-1 )
                Adj.append( ( ind00, ind0m ) )
    
    ## Build the adjacency matrix.
    AdjMatrix = sparse.coo_matrix( ( ones( len( Adj ) ), asarray( Adj ).T ), shape = ( rows*cols, rows*cols ) )
    #AdjMatrix = AdjMatrix.tocsr()
    
    ## Build the adjacency matrix representing cut edges and subtract it
    if len( cut_edges ) > 0:
        CutAdj = []
        for ij, kl in cut_edges:
            CutAdj.append( ( ij2ind( *ij ), ij2ind( *kl ) ) )
            CutAdj.append( ( ij2ind( *kl ), ij2ind( *ij ) ) )
        CutAdjMatrix = sparse.coo_matrix( ( ones( len( CutAdj ) ), asarray( CutAdj ).T ), shape = ( rows*cols, rows*cols ) )
        
        ## Update AdjMatrix
        AdjMatrix = AdjMatrix - CutAdjMatrix
    
    '''
    ## One over mass
    ooMass = sparse.identity( rows*cols )
    ooMass.setdiag( 1./asarray(AdjMatrix.sum(1)).ravel() )
    ## NOTE: ooMass*AdjMatrix isn't symmetric because of boundaries!!!
    L = sparse.identity( rows*cols ) - ooMass * AdjMatrix
    '''
    
    ## This formulation is symmetric: each vertex has a consistent weight
    ## according to its area (meaning boundary vertices have smaller
    ## weights than interior vertices).
    ## NOTE: I tried sparse.dia_matrix(), but sparse.dia_matrix.setdiag() fails with a statement that dia_matrix doesn't have element assignment.
    ## UPDATE: setdiag() seems to just be generally slow.  coo_matrix is fast!
    #Mass = sparse.lil_matrix( ( rows*cols, rows*cols ) )
    #Mass.setdiag( asarray(AdjMatrix.sum(1)).ravel() )
    #debugger()
    Mass = sparse.coo_matrix( ( asarray(AdjMatrix.sum(1)).ravel(), ( range( rows*cols ), range( rows*cols ) ) ) )
    L = .25 * ( Mass - AdjMatrix )
    
    #debugger()
    ## The rows should sum to 0.
    assert ( abs( asarray( L.sum(1) ).ravel() ) < 1e-5 ).all()
    ## The columns should also sum to 0, since L is symmetric.
    assert ( abs( asarray( L.sum(0) ).ravel() ) < 1e-5 ).all()
    ## It should be symmetric.
    assert len( ( L - L.T ).nonzero()[0] ) == 0
    
    return L

def gen_symmetric_grid_laplacian2( rows, cols, cut_edges = None ):
    '''
    The same as 'gen_symmetric_grid_laplacian1()', except boundary weights are correct.
    
    tested
    (see also test_cut_edges())
    '''
    
    assert rows > 0
    assert cols > 0
    
    if cut_edges is None: cut_edges = []
    
    assert_valid_cut_edges( rows, cols, cut_edges )
    
    from scipy import sparse
    
    N = rows
    M = cols
    def ind2ij( ind ):
        assert ind >= 0 and ind < N*M
        return ind // M, ind % M
    def ij2ind( i,j ):
        assert i >= 0 and i < N and j >= 0 and j < M
        return i*M + j
    
    Adj = []
    AdjValues = []
    
    ## The middle (lacking the first and last columns) strip down
    ## to the bottom, not including the bottom row.
    for i in xrange( 0, rows-1 ):
        for j in xrange( 1, cols-1 ):
            
            ind00 = ij2ind( i,j )
            indp0 = ij2ind( i+1,j )
            Adj.append( ( ind00, indp0 ) )
            AdjValues.append( .25 )
    
    ## The first and last columns down to the bottom,
    ## not including the bottom row.
    for i in xrange( 0, rows-1 ):
        for j in ( 0, cols-1 ):
            
            ind00 = ij2ind( i,j )
            indp0 = ij2ind( i+1,j )
            Adj.append( ( ind00, indp0 ) )
            AdjValues.append( .125 )
    
    ## The middle (lacking the first and last rows) strip to
    ## the right, not including the last column.
    for i in xrange( 1, rows-1 ):
        for j in xrange( 0, cols-1 ):
            
            ind00 = ij2ind( i,j )
            ind0p = ij2ind( i,j+1 )
            Adj.append( ( ind00, ind0p ) )
            AdjValues.append( .25 )
    
    ## The first and last rows over to the right,
    ## not including the right-most column.
    for i in ( 0, rows-1 ):
        for j in xrange( 0, cols-1 ):
            
            ind00 = ij2ind( i,j )
            ind0p = ij2ind( i,j+1 )
            Adj.append( ( ind00, ind0p ) )
            AdjValues.append( .125 )
    
    ## Build the adjacency matrix.
    AdjMatrix = sparse.coo_matrix( ( AdjValues, asarray( Adj ).T ), shape = ( rows*cols, rows*cols ) )
    ## We have so far only counted right and downward edges.
    ## Add left and upward edges by adding the transpose.
    AdjMatrix = AdjMatrix.T + AdjMatrix
    #AdjMatrix = AdjMatrix.tocsr()
    
    ## Build the adjacency matrix representing cut edges and subtract it
    if len( cut_edges ) > 0:
        CutAdj = []
        for ij, kl in cut_edges:
            CutAdj.append( ( ij2ind( *ij ), ij2ind( *kl ) ) )
            CutAdj.append( ( ij2ind( *kl ), ij2ind( *ij ) ) )
        CutAdjMatrix = sparse.coo_matrix( ( ones( len( CutAdj ) ), asarray( CutAdj ).T ), shape = ( rows*cols, rows*cols ) )
        
        ## Update AdjMatrix.
        ## We need to subtract the component-wise product of CutAdjMatrix and AdjMatrix
        ## because AdjMatrix has non-zero values and CutAdjMatrix acts like a mask.
        AdjMatrix = AdjMatrix - CutAdjMatrix.multiply( AdjMatrix )
    
    '''
    ## One over mass
    ooMass = sparse.identity( rows*cols )
    ooMass.setdiag( 1./asarray(AdjMatrix.sum(1)).ravel() )
    ## NOTE: ooMass*AdjMatrix isn't symmetric because of boundaries!!!
    L = sparse.identity( rows*cols ) - ooMass * AdjMatrix
    '''
    
    ## This formulation is symmetric: each vertex has a consistent weight
    ## according to its area (meaning boundary vertices have smaller
    ## weights than interior vertices).
    ## NOTE: I tried sparse.dia_matrix(), but sparse.dia_matrix.setdiag() fails with a statement that dia_matrix doesn't have element assignment.
    ## UPDATE: setdiag() seems to just be generally slow.  coo_matrix is fast!
    #Mass = sparse.lil_matrix( ( rows*cols, rows*cols ) )
    #Mass.setdiag( asarray(AdjMatrix.sum(1)).ravel() )
    #debugger()
    Mass = sparse.coo_matrix( ( asarray(AdjMatrix.sum(1)).ravel(), ( range( rows*cols ), range( rows*cols ) ) ) )
    L = ( Mass - AdjMatrix )
    
    ## The rows should sum to 0.
    assert ( abs( asarray( L.sum(1) ).ravel() ) < 1e-5 ).all()
    ## The columns should also sum to 0, since L is symmetric.
    assert ( abs( asarray( L.sum(0) ).ravel() ) < 1e-5 ).all()
    ## It should be symmetric.
    assert len( ( L - L.T ).nonzero()[0] ) == 0
    
    return L

def gen_grid_laplacian2_with_boundary_reflection( rows, cols, cut_edges = None ):
    '''
    The same as 'gen_symmetric_grid_laplacian2()',
    except the surface reflects across boundaries.
    The resulting matrix is not symmetric, however.
    
    tested
    (see also test_cut_edges())
    '''
    
    assert rows > 0
    assert cols > 0
    
    if cut_edges is None: cut_edges = []
    
    assert_valid_cut_edges( rows, cols, cut_edges )
    
    from scipy import sparse
    
    N = rows
    M = cols
    def ind2ij( ind ):
        assert ind >= 0 and ind < N*M
        return ind // M, ind % M
    def ij2ind( i,j ):
        assert i >= 0 and i < N and j >= 0 and j < M
        return i*M + j
    
    Adj = []
    AdjValues = []
    
    ## NOTE: Unlike gen_symmetric_grid_laplacian2(), we don't use different weights
    ##       for boundary edges, because we continue the surface across the boundary
    ##       by reflection.
    
    ## The middle strip down to the bottom, not including the bottom row.
    for i in xrange( 0, rows-1 ):
        for j in xrange( 0, cols ):
            
            ind00 = ij2ind( i,j )
            indp0 = ij2ind( i+1,j )
            Adj.append( ( ind00, indp0 ) )
            AdjValues.append( .25 )
    
    ## The middle strip to the right, not including the last column.
    for i in xrange( 0, rows ):
        for j in xrange( 0, cols-1 ):
            
            ind00 = ij2ind( i,j )
            ind0p = ij2ind( i,j+1 )
            Adj.append( ( ind00, ind0p ) )
            AdjValues.append( .25 )
    
    ## Build the adjacency matrix.
    AdjMatrix = sparse.coo_matrix( ( AdjValues, asarray( Adj ).T ), shape = ( rows*cols, rows*cols ) )
    ## We have so far only counted right and downward edges.
    ## Add left and upward edges by adding the transpose.
    AdjMatrix = AdjMatrix.T + AdjMatrix
    #AdjMatrix = AdjMatrix.tocsr()
    
    ## Build the adjacency matrix representing cut edges and subtract it
    CutAdj = []
    if len( cut_edges ) > 0:
        def reflect_a_across_b( a, b ):
            ## What we want is the step from a to b taken from b:
            ## ( b - a ) + b = 2*b - a
            ## NOTE: The caller of this function may be assuming
            ##       that a and b are adjacent (horizontally or vertically);
            ##       this was checked at the beginning of gen_grid_laplacian*().
            return ( 2*b[0] - a[0], 2*b[1] - a[1] )
        def ij_valid( ij ):
            return ij[0] >= 0 and ij[1] >= 0 and ij[0] < rows and ij[1] < cols
        
        for ij, kl in cut_edges:
            CutAdj.append( ( ij2ind( *ij ), ij2ind( *kl ) ) )
            ## Boundary reflection works out to simply cutting
            ## the edge between ij and the reflection of kl across ij
            ## from ij's point of view.
            ## Our matrix will no longer be symmetric :-(.
            kl_across_ij = reflect_a_across_b( kl, ij )
            if ij_valid( kl_across_ij ):
                CutAdj.append( ( ij2ind( *ij ), ij2ind( *kl_across_ij ) ) )
            
            CutAdj.append( ( ij2ind( *kl ), ij2ind( *ij ) ) )
            ## Boundary reflection works out to simply cutting
            ## the edge between kl and the reflection of ij across kl
            ## from kl's point of view.
            ## Our matrix will no longer be symmetric :-(.
            ij_across_kl = reflect_a_across_b( ij, kl )
            if ij_valid( ij_across_kl ):
                CutAdj.append( ( ij2ind( *kl ), ij2ind( *ij_across_kl ) ) )
    
    ## We also add "cut_edges" for the grid boundary.
    if rows > 1:
        for j in xrange( 0, cols ):
            CutAdj.append( ( ij2ind( 0, j ), ij2ind( 1, j ) ) )
            CutAdj.append( ( ij2ind( rows-1, j ), ij2ind( rows-2, j ) ) )
    if cols > 1:
        for i in xrange( 0, rows ):
            CutAdj.append( ( ij2ind( i, 0 ), ij2ind( i, 1 ) ) )
            CutAdj.append( ( ij2ind( i, cols-1 ), ij2ind( i, cols-2 ) ) )
    
    if len( CutAdj ) > 0:
        ## Cutting reflected edges could lead to some entries appearing multiple times.
        ## We want to use the entries like a {0,1} mask,
        ## so make sure each entry appears at most once.
        CutAdj = list(set( CutAdj ))
        
        CutAdjMatrix = sparse.coo_matrix( ( ones( len( CutAdj ) ), asarray( CutAdj ).T ), shape = ( rows*cols, rows*cols ) )
        
        ## Update AdjMatrix.
        ## We need to subtract the component-wise product of CutAdjMatrix and AdjMatrix
        ## because AdjMatrix has non-zero values and CutAdjMatrix acts like a mask.
        AdjMatrix = AdjMatrix - CutAdjMatrix.multiply( AdjMatrix )
    
    '''
    ## One over mass
    ooMass = sparse.identity( rows*cols )
    ooMass.setdiag( 1./asarray(AdjMatrix.sum(1)).ravel() )
    ## NOTE: ooMass*AdjMatrix isn't symmetric because of boundaries!!!
    L = sparse.identity( rows*cols ) - ooMass * AdjMatrix
    '''
    
    ## This formulation is symmetric: each vertex has a consistent weight
    ## according to its area (meaning boundary vertices have smaller
    ## weights than interior vertices).
    ## NOTE: I tried sparse.dia_matrix(), but sparse.dia_matrix.setdiag() fails with a statement that dia_matrix doesn't have element assignment.
    ## UPDATE: setdiag() seems to just be generally slow.  coo_matrix is fast!
    #Mass = sparse.lil_matrix( ( rows*cols, rows*cols ) )
    #Mass.setdiag( asarray(AdjMatrix.sum(1)).ravel() )
    #debugger()
    Mass = sparse.coo_matrix( ( asarray(AdjMatrix.sum(1)).ravel(), ( range( rows*cols ), range( rows*cols ) ) ) )
    L = ( Mass - AdjMatrix )
    
    ## The rows should sum to 0.
    assert ( abs( asarray( L.sum(1) ).ravel() ) < 1e-5 ).all()
    ## The columns need not also sum to 0, since L is not symmetric.
    #assert ( abs( asarray( L.sum(0) ).ravel() ) < 1e-5 ).all()
    
    ## No row should be empty.
    ## UPDATE: The corner vertices will be empty.
    #assert ( asarray( abs( L ).sum(1) ).ravel() > 1e-5 ).all()
    ## UPDATE 2: With cut edges we can have an arbitrary number of
    ##           corner-like vertices.
    assert len( where( asarray( abs( L ).sum(1) ).ravel() < 1e-5 )[0] ) == ( 2 if rows == 1 or cols == 1 else 4 ) or len( cut_edges ) > 0
    ## UPDATE 2: But their columns should still have non-zero values.
    assert ( asarray( abs( L ).sum(0) ).ravel() > 1e-5 ).all()
    
    ## It should be symmetric.
    #assert len( ( L - L.T ).nonzero()[0] ) == 0
    
    return L

gen_symmetric_grid_laplacian = gen_symmetric_grid_laplacian2

def gen_constraint_system( rows, cols, dx_constraints = [], dy_constraints = [], value_constraints = [] ):
    '''
    Given dimensions of a 'rows' by 'cols' 2D grid,
    a sequence 'dx_constraints' of tuples ( i, j, dx ) specifying the value of grid[ i+1,j ] - grid[ i,j ],
    a sequence 'dy_constraints' of tuples ( i, j, dy ) specifying the value of grid[ i,j+1 ] - grid[ i,j ],
    a sequence 'value_constraints' of tuples ( i, j, val ) specifying the value of grid[ i,j ],
    returns two items, a matrix 'C' and an array 'Crhs', representing the constraints specified
    by 'dx_constraints' and 'dy_constraints' and 'value_constraints' in matrix form:
        C*dof = Crhs
    
    NOTE: The constrained values dx, dy, and val can be K-dimensional vectors instead of scalars,
          in which case Crhs will have shape #constraints-by-K.
    
    tested
    '''
    
    from scipy import sparse
    
    ## Indices for dx constraints should be valid.
    assert False not in [ i >= 0 and i < rows-1 for i, j, dx in dx_constraints ]
    assert False not in [ j >= 0 and j < cols for i, j, dx in dx_constraints ]
    
    ## Indices for dy constraints should be valid.
    assert False not in [ i >= 0 and i < rows for i, j, dy in dy_constraints ]
    assert False not in [ j >= 0 and j < cols-1 for i, j, dy in dy_constraints ]
    
    ## Indices for value constraints should be valid.
    assert False not in [ i >= 0 and i < rows for i, j, val in value_constraints ]
    assert False not in [ j >= 0 and j < cols for i, j, val in value_constraints ]
    
    ## There shouldn't be duplicate constraints.
    assert len( set( [ (i,j) for i, j, dx in dx_constraints ] ) ) == len( dx_constraints )
    assert len( set( [ (i,j) for i, j, dy in dy_constraints ] ) ) == len( dy_constraints )
    assert len( set( [ (i,j) for i, j, val in value_constraints ] ) ) == len( value_constraints )
    
    def ij2ind( i,j ):
        assert i >= 0 and i < rows and j >= 0 and j <= cols
        return i*cols + j
    
    ### Constrain gradients (system).
    ## The following three lists define linear equations: \sum_i \sum_j zs[ indices[i][j] ] * vals[i][j] = rhs[i]
    indices = []
    vals = []
    rhs = []
    ## Do this by constraining directional differences according to the given dx and dy in 'dx_constraints' and 'dy_constraints'.
    for i,j,dx in dx_constraints:
        indices.append( [ ij2ind( i+1, j ), ij2ind( i, j ) ] )
        vals.append( [ 1., -1. ] )
        rhs.append( dx )
    
    for i,j,dy in dy_constraints:
        indices.append( [ ij2ind( i, j+1 ), ij2ind( i, j ) ] )
        vals.append( [ 1., -1. ] )
        rhs.append( dy )
    
    for i,j,val in value_constraints:
        indices.append( [ ij2ind( i, j ) ] )
        vals.append( [ 1. ] )
        rhs.append( val )
    
    assert len( indices ) == len( vals )
    assert len( indices ) == len( rhs )
    
    ## Build the constraints system.
    Crows = [ [i] * len( js ) for i, js in enumerate( indices ) ]
    #debugger()
    C = sparse.coo_matrix( ( concatenate( vals ), ( concatenate( Crows ), concatenate( indices ) ) ), shape = ( len( Crows ), rows*cols ) )
    Crhs = asarray( rhs )
    
    return C, Crhs

def gen_constraint_rhs( dx_constraints = [], dy_constraints = [], value_constraints = [] ):
    '''
    Given a sequence 'dx_constraints' of values dx,
    a sequence 'dy_constraints' of values dy,
    a sequence 'value_constraints' of values val,
    returns an array 'Crhs', representing the right-hand-side for the system generated by
    gen_constraint_system() with same-named 'dx_constraints' and 'dy_constraints' and
    'value_constraints' in matrix form:
        C*dof = Crhs
    
    NOTE: The constrained values dx, dy, and val can be K-dimensional vectors instead of scalars,
          in which case Crhs will have shape #constraints-by-K.
    '''
    
    Crhs = asarray( list( dx_constraints ) + list( dy_constraints ) + list( value_constraints ) )
    return Crhs

def system_and_rhs_with_value_constraints1( system, rhs, value_constraints, cols ):
    '''
    tested
    '''
    
    from tictoc import tic, toc
    
    ## Copy rhs, because we modify it in-place.
    rhs = array( rhs )
    
    tic( 'value constraint walking:' )
    
    ## Now incorporate value_constraints by setting some identity rows and changing the right-hand-side.
    ## Set constraint rows to identity rows.
    ## We also zero the columns to keep the matrix symmetric.
    ## We also have to update the right-hand-side when we zero the columns.
    
    ## There shouldn't be duplicate constraints.
    assert len( set( [ (i,j) for i, j, val in value_constraints ] ) ) == len( value_constraints )
    
    ## A dict has the right algorithmic access properties for what we want,
    ## which is a lot of checking whether a given degree-of-freedom is inside.
    value_constraints_dict = dict([ ( i*cols+j, val ) for i,j,val in value_constraints ])
    
    ## Turn 'system' into a coo_matrix so it has .data, .row, .col properties.
    system = system.tocoo()
    
    ## Set corresponding entries of the right-hand-side vector to the constraint value.
    for index, value in value_constraints_dict.iteritems(): rhs[ index ] = value
    
    ## Update the right-hand-side elements wherever a fixed degree-of-freedom
    ## is involved in a row (the columns of constrained degrees-of-freedom).
    for val, row, col in zip( system.data, system.row, system.col ):
        if row in value_constraints_dict and col not in value_constraints_dict:
            rhs[ col ] -= val * value_constraints_dict[ row ]
    
    system_vij = [ (val,row,col) for val,row,col in zip( system.data, system.row, system.col ) if row not in value_constraints_dict and col not in value_constraints_dict ]
    ## Then add a 1 along the diagonal.
    system_vij.extend( [ ( 1., index, index ) for index in value_constraints_dict.iterkeys() ] )
    
    toc()
    tic( 'transpose vij:' )
    
    ## Transpose system_vij for passing to matrix constructors.
    ## UPDATE: Also convert each row to an array so that cvxopt doesn't complain.
    system_vij = [ asarray( a ) for a in zip( *system_vij ) ]
    
    del system
    del value_constraints_dict
    toc()
    
    import scipy.sparse
    system = scipy.sparse.coo_matrix( ( system_vij[0], system_vij[1:] ) )
    return system, rhs

def system_and_rhs_with_value_constraints2( system, rhs, value_constraints, cols ):
    '''
    tested
    '''
    
    from scipy import sparse
    
    ## Now incorporate value_constraints by setting some identity rows and changing the right-hand-side.
    ## Set constraint rows to identity rows.
    ## We also zero the columns to keep the matrix symmetric.
    ## NOTE: We have to update the right-hand-side when we zero the columns.
    
    ## There shouldn't be duplicate constraints.
    assert len( set( [ (i,j) for i, j, val in value_constraints ] ) ) == len( value_constraints )
    
    value_constraint_values = asarray([ val for i,j,val in value_constraints ])
    value_constraint_indices = asarray([ (i,j) for i,j,val in value_constraints ])
    value_constraint_indices = value_constraint_indices[:,0] * cols + value_constraint_indices[:,1]
    
    ## Update the right-hand-side elements wherever a fixed degree-of-freedom
    ## is involved in a row (the columns of constrained degrees-of-freedom).
    '''
    for index, val in zip( value_constraint_values, value_constraint_indices ):
        rhs -= system.getrow( index ) * val
    '''
    #R = sparse.csr_matrix( ( 1, system.shape[0] ) )
    #for index, val in zip( value_constraint_values, value_constraint_indices ): R[ 0, index ] = val
    R = sparse.coo_matrix(
        ## values
        ( value_constraint_values,
        ## row indices are zero
        ( zeros( value_constraint_indices.shape, value_constraint_indices.dtype ),
        ## column indices
        value_constraint_indices ) ),
        ## shape is 1 x N
        shape = ( 1, system.shape[0] )
        )
    R = R.tocsr()
    rhs = rhs - R * system
    rhs = asarray( rhs ).ravel()
    
    ## Set corresponding entries of the right-hand-side vector to the constraint value.
    #for index, value in value_constraints_dict.iteritems(): rhs[ index ] = value
    rhs[ value_constraint_indices ] = value_constraint_values
    
    
    ## Zero the constrained rows and columns, and set the diagonal to 1.
    
    ## Zero the rows and columns.
    diag = ones( rhs.shape[0] )
    diag[ value_constraint_indices ] = 0.
    #D = sparse.identity( system.shape[0] )
    #D.setdiag( diag )
    D = sparse.coo_matrix( ( diag, ( arange( system.shape[0] ), arange( system.shape[0] ) ) ) ).tocsr()
    system = D * system * D
    
    ## Set the constraints' diagonals to 1.
    #diag = zeros( rhs.shape[0] )
    #diag[ value_constraint_indices ] = 1.
    #D.setdiag( diag )
    #D.setdiag( 1. - diag )
    D = sparse.coo_matrix( ( ones( value_constraint_indices.shape[0] ), ( value_constraint_indices, value_constraint_indices ) ), shape = system.shape )
    system = system + D
    
    return system, rhs

def system_and_rhs_with_value_constraints2_multiple_rhs( system, rhs, value_constraints, cols ):
    '''
    used (successfully by a function that is tested)
    '''
    
    from scipy import sparse
    
    ## Now incorporate value_constraints by setting some identity rows and changing the right-hand-side.
    ## Set constraint rows to identity rows.
    ## We also zero the columns to keep the matrix symmetric.
    ## NOTE: We have to update the right-hand-side when we zero the columns.
    
    ## There shouldn't be duplicate constraints.
    assert len( set( [ (i,j) for i, j, val in value_constraints ] ) ) == len( value_constraints )
    
    value_constraint_values = asarray([ val for i,j,val in value_constraints ])
    value_constraint_indices = asarray([ (i,j) for i,j,val in value_constraints ])
    value_constraint_indices = value_constraint_indices[:,0] * cols + value_constraint_indices[:,1]
    
    ## Update the right-hand-side elements wherever a fixed degree-of-freedom
    ## is involved in a row (the columns of constrained degrees-of-freedom).
    '''
    for index, val in zip( value_constraint_values, value_constraint_indices ):
        rhs -= system.getrow( index ) * val
    '''
    #R = sparse.csr_matrix( ( 1, system.shape[0] ) )
    #for index, val in zip( value_constraint_values, value_constraint_indices ): R[ 0, index ] = val
    R = sparse.coo_matrix(
        ## values
        ( value_constraint_values.T.ravel(),
        ## row indices are zeros for the zero-th element in each value, ones for the one-th element in each value, and so on.
        ( repeat( arange( value_constraint_values.shape[1], dtype = value_constraint_indices.dtype ), value_constraint_indices.shape[0] ),
        ## column indices
        tile( value_constraint_indices, value_constraint_values.shape[1] ) ) ),
        ## shape is value_constraint_values.shape[1] x N
        shape = ( value_constraint_values.shape[1], system.shape[0] )
        )
    R = R.tocsr()
    rhs = rhs - ( R * system ).T
    rhs = asarray( rhs )
    
    ## Set corresponding entries of the right-hand-side vector to the constraint value.
    #for index, value in value_constraints_dict.iteritems(): rhs[ index ] = value
    rhs[ value_constraint_indices, : ] = value_constraint_values
    
    
    ## Zero the constrained rows and columns, and set the diagonal to 1.
    
    ## Zero the rows and columns.
    diag = ones( rhs.shape[0] )
    diag[ value_constraint_indices ] = 0.
    #D = sparse.identity( system.shape[0] )
    #D.setdiag( diag )
    D = sparse.coo_matrix( ( diag, ( arange( system.shape[0] ), arange( system.shape[0] ) ) ) ).tocsr()
    system = D * system * D
    
    ## Set the constraints' diagonals to 1.
    #diag = zeros( rhs.shape[0] )
    #diag[ value_constraint_indices ] = 1.
    #D.setdiag( diag )
    #D.setdiag( 1. - diag )
    D = sparse.coo_matrix( ( ones( value_constraint_indices.shape[0] ), ( value_constraint_indices, value_constraint_indices ) ), shape = system.shape )
    system = system + D
    
    return system, rhs

## Use 'system_and_rhs_with_value_constraints2', since it is much faster.
system_and_rhs_with_value_constraints = system_and_rhs_with_value_constraints2


def system_with_value_constraints3( system, value_constraints, cols ):
    '''
    Like system_and_rhs_with_value_constraints2_multiple_rhs()
    except value_constraints are only (i,j) tuples and the resulting system
    does not stay symmetric. This is so that when the right-hand side changes, we don't
    have to change the system matrix.
    '''
    
    from scipy import sparse
    
    ## Now incorporate value_constraints by setting some identity rows and changing the right-hand-side.
    ## Set constraint rows to identity rows.
    
    ## There shouldn't be duplicate constraints.
    assert len( set( value_constraints ) ) == len( value_constraints )
    value_constraint_indices = asarray( value_constraints )
    value_constraint_indices = value_constraint_indices[:,0] * cols + value_constraint_indices[:,1]
    
    ## Zero the rows.
    diag = ones( system.shape[0] )
    diag[ value_constraint_indices ] = 0.
    #D = sparse.identity( system.shape[0] )
    #D.setdiag( diag )
    D = sparse.coo_matrix( ( diag, ( arange( system.shape[0] ), arange( system.shape[0] ) ) ) ).tocsr()
    system = D * system
    
    ## Set the constraints' diagonals to 1.
    D = sparse.coo_matrix( ( ones( value_constraint_indices.shape[0] ), ( value_constraint_indices, value_constraint_indices ) ), shape = system.shape )
    system = system + D
    
    return system

def rhs_with_value_constraints3( rhs, value_constraint_indices, value_constraint_values, cols ):
    '''
    Like system_and_rhs_with_value_constraints2_multiple_rhs()
    except the value constraints are passed as
    a sequence of (i,j) 'value_constraint_indices' and
    a sequence of scalar or k-dimensional values 'value_constraints_values'.
    '''
    
    ## Now incorporate value_constraints by setting some identity rows and changing the right-hand-side.
    ## Set constraint rows to identity rows.
    
    ## There shouldn't be duplicate constraints.
    ## Convert value_constraint_indices entries to tuples so they can be added to a set().
    assert len( set([ (i,j) for i,j in value_constraint_indices ]) ) == len( value_constraint_indices )
    ## We should have the same number of values as indices.
    assert len( value_constraint_indices ) == len( value_constraint_values )
    value_constraint_indices = asarray( value_constraint_indices )
    value_constraint_indices = value_constraint_indices[:,0] * cols + value_constraint_indices[:,1]
    
    rhs = array( rhs )
    rhs[ value_constraint_indices ] = value_constraint_values
    return rhs

def solve_grid_linear( rows, cols, dx_constraints = None, dy_constraints = None, value_constraints = None, bilaplacian = False, iterative = False, cut_edges = None, w_lsq = None ):
    '''
    Given dimensions of a 'rows' by 'cols' 2D grid,
    an optional sequence 'dx_constraints' of tuples ( i, j, dx ) specifying the value of grid[ i+1,j ] - grid[ i,j ],
    an optional sequence 'dy_constraints' of tuples ( i, j, dy ) specifying the value of grid[ i,j+1 ] - grid[ i,j ],
    an optional sequence 'value_constraints' of tuples ( i, j, value ) specifying the value of grid[ i,j ],
    returns the solution to the laplace equation on the domain given by 'mesh' with derivatives
    given by 'dx_constraints' and 'dy_constraints' and values given by 'value_constraints'.
    
    If 'bilaplacian' is true, this will solve bilaplacian f = 0, and boundaries will be reflected.
    
    If 'iterative' is true, an iterative solver will be used instead of a direct method.
    
    
    Optional parameter 'cut_edges' specifies edges that should be disconnected
    in the laplacian/bilaplacian system.  The parameter must be in the format
    suitable for passing to 'gen_symmetric_grid_laplacian().'
    The returned solution is almost guaranteed to be discontinuous across the edges.
    The convenience function 'cut_edges_from_mask()' can be used to create
    the 'cut_edges' parameter from a mask.
    To allow derivative discontinuity but still require connectedness across edges,
    places those edges into 'dx_constraints' and 'dy_constraints' with
    constraint value 0.  The convenience function
    'zero_dx_dy_constraints_from_cut_edges()' can be used for this.
    
    Optional parameter 'w_lsq' determines the weight for the
    least-squares constraints (dx/dy_constraints).
    
    tested
    (see also test_cut_edges())
    '''
    
    if dx_constraints is None: dx_constraints = []
    if dy_constraints is None: dy_constraints = []
    if value_constraints is None: value_constraints = []
    
    ## Smoothness term.
    w_smooth = 1.
    #w_smooth = 10.
    ## Constraints (normals and z0 = 0) term.
    if w_lsq is None:
        w_gradients = 1e5
        ## Reconstruct with more weight on the smoothing:
        #w_gradients = 1e-1
    else:
        assert float( w_lsq ) == w_lsq
        w_gradients = float( w_lsq )
    
    print 'solve_grid_linear( rows = %s, cols = %s, |constraints| = %s, iterative = %s, bilaplacian = %s, |cut_edges| = %s, w_lsq = %s )' % (
        rows, cols,
        len( dx_constraints ) + len( dy_constraints ) + len( value_constraints ),
        iterative,
        bilaplacian,
        len( cut_edges ) if cut_edges is not None else None,
        w_gradients
        )
    
    ## The grid can get large; splitting the building of L and C into seperate functions
    ## means that the temporaries go away as soon as we build the matrices.
    from tictoc import tic, toc
    tic( 'build energy:' )
    
    #L = gen_symmetric_grid_laplacian1( rows, cols, cut_edges )
    #L = gen_symmetric_grid_laplacian2( rows, cols, cut_edges )
    ## NOTE: I would always call gen_grid_laplacian2_with_boundary_reflection(),
    ##       but the matrices are too large to solve with LU decomposition,
    ##       our unsymmetric solver.  This isn't a problem with the bilaplacian,
    ##       because we multiply the matrix by its transpose, which
    ##       always results in a symmetric matrix.
    #debugger()
    ## Bi-Laplacian instead of Laplacian.
    if bilaplacian:
        L = gen_grid_laplacian2_with_boundary_reflection( rows, cols, cut_edges )
        L = L.T*L
    else:
        L = gen_symmetric_grid_laplacian( rows, cols, cut_edges )
    
    toc()
    tic( 'add constraints:' )
    tic( 'dx dy constraints:' )
    
    if len( dx_constraints ) + len( dy_constraints ) > 0:
        C, Crhs = gen_constraint_system( rows, cols, dx_constraints, dy_constraints, [] if ( 0 < len(value_constraints) ) else [ (0,0,0.) ] )
        
        #debugger()
        ## The system to solve:
        ## w_smooth * ( L * grid.ravel() ) + w_gradients * ( C.T * C * grid.ravel() ) = w_gradients * C.T * Crhs
        
        system = ( w_smooth * L + ( w_gradients * C.T ) * C )
        rhs = ( ( w_gradients * C.T ) * Crhs )
        
        del C
        del Crhs
    
    else:
        system = w_smooth * L
        rhs = zeros( L.shape[0] )
    
    del L
    
    toc()
    
    tic( 'value constraints (fast):' )
    if len( value_constraints ) > 0:
        #system_grads = system
        system, rhs = system_and_rhs_with_value_constraints2( system, rhs, value_constraints, cols )
    toc()
    
    '''
    tic( 'value constraints (slow):' )
    s2, r2 = system_and_rhs_with_value_constraints1( system, rhs, value_constraints, cols )
    toc()
    ## We pass these tests.
    assert ( system - s2 ).nnz == 0
    assert len( where( rhs - r2 )[0] ) == 0
    '''
    
    toc()
    
    if iterative:
        ## Iterative solution.
        import scipy.sparse
        tic( 'convert matrix for solving' )
        
        #system = scipy.sparse.coo_matrix( ( system_vij[0], system_vij[1:] ) ).tocsr()
        #del system_vij
        
        system = system.tocsr()
        
        toc()
        import scipy.sparse.linalg
        tic( 'solve system of equations cg:' )
        result, info = scipy.sparse.linalg.cg( system, rhs )
        toc()
        assert 0 == info
        result = array( result ).ravel()
    else:
        import cvxopt, cvxopt.cholmod, cvxopt.umfpack
        tic( 'convert matrix for solving:' )
        system = system.tocoo()
        
        ## Q: Why does cvxopt compiled on Mac OS X 10.6 instead of 10.5 complains about invalid array types?
        ## A: Because it insists on having the row and col parameters with type 'int', not 'int32'.
        #system = cvxopt.spmatrix( system.data, system.row, system.col )
        #system_scipy = system
        system = cvxopt.spmatrix( system.data, asarray( system.row, dtype = int ), asarray( system.col, dtype = int ) )
        
        #system = cvxopt.spmatrix( system_vij[0], system_vij[1] , system_vij[2] )
        #system = cvxopt.spmatrix( *zip( *system_vij ) )
        #del system_vij
        toc()
        rhs = cvxopt.matrix( rhs )
        
        tic( 'solve system of equations direct:' )
        ## Works
        cvxopt.cholmod.linsolve( system, rhs )
        ## Eats all memory then dies:
        #cvxopt.umfpack.linsolve( system, rhs )
        toc()
        result = array( rhs ).ravel()
        
        ## Eats all memory then dies (based on SuperLU):
        #import scipy.sparse as sparse
        #import scipy.sparse.linalg
        #scipy.sparse.linalg.use_solver( useUmfpack = True )
        #result = sparse.linalg.factorized( w_smooth * L + ( w_gradients * C.T ) * C )( ( w_gradients * C.T ) * Crhs ).reshape( (rows, cols) )
        
        ## Eats all memory then dies (based on SuperLU):
        #import scipy.sparse.linalg
        #result = sparse.linalg.spsolve( w_smooth * L + ( w_gradients * C.T ) * C, ( w_gradients * C.T ) * Crhs ).reshape( (rows, cols) )
    
    return result.reshape( ( rows, cols ) )

def solve_grid_linear_simple( rows, cols, value_constraints, bilaplacian = False ):
    '''
    Given dimensions of a 'rows' by 'cols' 2D grid,
    a sequence 'value_constraints' of tuples ( i, j, [ value1, value2, ... ] ) specifying the n-dimensional values of grid[ i,j ],
    returns the solution to the laplace equation on the domain given by 'mesh'
    with values given by 'value_constraints'.
    
    If 'bilaplacian' is true, this will solve bilaplacian f = 0, and boundaries will be reflected.
    
    tested (see end of function); should be equivalent to:
    return dstack([
        solve_grid_linear( rows, cols, value_constraints = [ (i,j,vals[c]) for (i,j,vals) in value_constraints ], bilaplacian = bilaplacian )
        for c in xrange( list(set([len(vals) for (i,j,vals) in value_constraints]))[0] )
        ])
    '''
    
    print 'solve_grid_linear_simple( rows = %s, cols = %s, |constraints| = %s, bilaplacian = %s )' % (
        rows, cols,
        len( value_constraints ),
        bilaplacian
        )
    
    from tictoc import tic, toc
    tic( 'build energy:' )
    
    ## Bi-Laplacian instead of Laplacian.
    if bilaplacian:
        L = gen_grid_laplacian2_with_boundary_reflection( rows, cols )
        L = L.T*L
    else:
        L = gen_symmetric_grid_laplacian( rows, cols )
    
    toc()
    tic( 'add constraints:' )
    
    system = L
    del L
    
    ## We must have at least one value constraint or our system is under-constrained.
    assert len( value_constraints ) > 0
    ## All value constraint values should be vectors of length >= 1.
    assert all([ len( asarray( vals ).shape ) == 1 for ( i,j,vals ) in value_constraints ])
    ## All value constraint values should be vectors with the same length.
    result_dim = list( set([ len( vals ) for ( i,j,vals ) in value_constraints ]) )
    assert len( result_dim ) == 1
    result_dim = result_dim[0]
    
    tic( 'value constraints (fast):' )
    system, rhs = system_and_rhs_with_value_constraints2_multiple_rhs( system, zeros( ( system.shape[0], result_dim ) ), value_constraints, cols )
    toc()
    
    toc()
    
    import cvxopt, cvxopt.cholmod, cvxopt.umfpack
    tic( 'convert matrix for solving:' )
    system = system.tocoo()
    
    system = cvxopt.spmatrix( system.data, asarray( system.row, dtype = int ), asarray( system.col, dtype = int ) )
    
    toc()
    
    rhs = cvxopt.matrix( rhs )
    
    tic( 'solve system of equations direct:' )
    cvxopt.cholmod.linsolve( system, rhs )
    toc()
    
    result = array( rhs ).ravel().reshape( ( rows, cols, result_dim ) )
    
    ## We pass this test:
    '''
    assert allclose( result, dstack([
        solve_grid_linear( rows, cols, value_constraints = [ (i,j,vals[c]) for (i,j,vals) in value_constraints ], bilaplacian = bilaplacian )
        for c in xrange( list(set([len(vals) for (i,j,vals) in value_constraints]))[0] )
        ]) )
    '''
    
    return result

def solve_grid_linear_simple2( rows, cols, value_constraints_hard = [], value_constraints_soft = [], w_lsq = 1., bilaplacian = False ):
    '''
    Given dimensions of a 'rows' by 'cols' 2D grid,
    a sequence 'value_constraints_hard' of tuples ( i, j, [ value1, value2, ... ] ) specifying the n-dimensional values of grid[ i,j ] as a hard constraint,
    a sequence 'value_constraints_soft' of tuples ( i, j, [ value1, value2, ... ] ) specifying the n-dimensional values of grid[ i,j ] as a soft (least squares) constraint,
    a parameter for the weight of all soft constraints 'w_lsq',
    returns the solution to the laplace equation on the domain given by 'mesh'
    with values given by 'value_constraints_hard', 'values_constraints_soft', and 'w_lsq'.
    
    If 'bilaplacian' is true, this will solve bilaplacian f = 0, and boundaries will be reflected.
    If 'w_lsq' is a sequence instead of a scalar, it must have the same length as 'value_constraints_soft' and will be used to constrain each equation.
    
    tested
    '''
    
    print 'solve_grid_linear_simple2( rows = %s, cols = %s, |hard constraints| = %s, |soft constraints| = %s, w_lsq = %s, bilaplacian = %s )' % (
        rows, cols,
        len( value_constraints_hard ),
        len( value_constraints_soft ),
        w_lsq if not hasattr( w_lsq, '__iter__' ) else '%s ...' % w_lsq[:3],
        bilaplacian
        )
    
    from tictoc import tic, toc
    tic( 'build energy:' )
    
    ## Bi-Laplacian instead of Laplacian.
    if bilaplacian:
        L = gen_grid_laplacian2_with_boundary_reflection( rows, cols )
        L = L.T*L
    else:
        L = gen_symmetric_grid_laplacian( rows, cols )
    
    toc()
    tic( 'add constraints:' )
    
    system = L
    del L
    
    ## We must have at least one value constraint or our system is under-constrained.
    assert len( value_constraints_hard ) + len( value_constraints_soft ) > 0
    ## All value constraint values should be vectors of length >= 1.
    assert all([ len( asarray( vals ).shape ) == 1 for ( i,j,vals ) in itertools.chain( value_constraints_hard, value_constraints_soft ) ])
    ## All value constraint values should be vectors with the same length.
    result_dim = list( set([ len( vals ) for ( i,j,vals ) in itertools.chain( value_constraints_hard, value_constraints_soft ) ]) )
    assert len( result_dim ) == 1
    result_dim = result_dim[0]
    
    ## Initialize the right-hand-side
    rhs = zeros( ( system.shape[0], result_dim ) )
    
    ## Least squares weight should be non-negative:
    assert w_lsq >= 0.
    
    tic( 'value constraints soft:' )
    if len( value_constraints_soft ) > 0:
        C, Crhs = gen_constraint_system( rows, cols, value_constraints = value_constraints_soft )
        
        #debugger()
        ## The system to solve:
        ## w_smooth * ( L * grid.ravel() ) + w_gradients * ( C.T * C * grid.ravel() ) = w_gradients * C.T * Crhs
        
        ## Is w_lsq iterable? If so, then we have a weight per soft constraint equation.
        if hasattr( w_lsq, '__iter__' ):
            w_lsq = asfarray( w_lsq )
            assert len( w_lsq ) == len( Crhs )
        else:
            w_lsq = w_lsq * ones( len( Crhs ) )
        
        from scipy import sparse
        W = sparse.coo_matrix( ( w_lsq, ( range( len( w_lsq ) ), range( len( w_lsq ) ) ) ) )
        
        CTW = C.T*W
        system = ( system + CTW*C )
        rhs = ( rhs + CTW*Crhs )
        
        del C
        del CTW
        del W
        del Crhs
    toc()
    
    ## NOTE: We must set hard constraints *after* we set soft constraints, because
    ##       the hard constraints set certain rows and columns to identity rows and columns.
    tic( 'value constraints hard (fast):' )
    if len( value_constraints_hard ) > 0:
        system, rhs = system_and_rhs_with_value_constraints2_multiple_rhs( system, rhs, value_constraints_hard, cols )
    toc()
    
    toc()
    
    import cvxopt, cvxopt.cholmod, cvxopt.umfpack
    tic( 'convert matrix for solving:' )
    system = system.tocoo()
    
    system = cvxopt.spmatrix( system.data, asarray( system.row, dtype = int ), asarray( system.col, dtype = int ) )
    
    toc()
    
    rhs = cvxopt.matrix( rhs )
    
    tic( 'solve system of equations direct:' )
    cvxopt.cholmod.linsolve( system, rhs )
    toc()
    
    result = array( rhs ).ravel().reshape( ( rows, cols, result_dim ) )
    
    ## We pass this test (with default epsilon with hard constraints, and within 1e-4 when w_lsq = 1e3):
    #assert allclose( result, solve_grid_linear_simple( rows, cols, list( itertools.chain( value_constraints_hard, value_constraints_soft ) ), bilaplacian = bilaplacian ) )
    
    return result

def solve_grid_linear_simple3_solver( rows, cols, value_constraints_hard = [], value_constraints_soft = [], w_lsq = 1., bilaplacian = False ):
    '''
    Given dimensions of a 'rows' by 'cols' 2D grid,
    a sequence 'value_constraints_hard' of tuples ( i, j ) specifying grid[ i,j ] as a hard constraint (NOTE: no values),
    a sequence 'value_constraints_soft' of tuples ( i, j ) specifying grid[ i,j ] as a soft (least squares) constraint (NOTE: no values),
    a parameter for the weight of all soft constraints 'w_lsq',
    returns a function solve( hard_values = [], soft_values = [] ) that can be used to efficiently
    compute multiple solutions to the laplace equation on the domain given by 'mesh' with different values
    for the constrained grid locations specified by 'value_constraints_hard' and 'value_constraints_soft'.
    The 'hard_values' and 'soft_values' parameters to the returned solve() function
    must have the same length as the 'value_constraints_hard' and 'value_constraints_soft' parameters.
    Each element of 'hard_values' is the value for the grid location specified by the corresponding element of 'value_constraints_hard'.
    Each element of 'soft_values' is the value for the grid location specified by the corresponding element of 'value_constraints_soft'.
    
    If 'bilaplacian' is true, this will solve bilaplacian f = 0, and boundaries will be reflected.
    If 'w_lsq' is a sequence instead of a scalar, it must have the same length as 'value_constraints_soft' and will be used to constrain each equation.
    
    tested
    '''
    
    print 'solve_grid_linear_simple2( rows = %s, cols = %s, |hard constraints| = %s, |soft constraints| = %s, w_lsq = %s, bilaplacian = %s )' % (
        rows, cols,
        len( value_constraints_hard ),
        len( value_constraints_soft ),
        w_lsq if not hasattr( w_lsq, '__iter__' ) else '%s ...' % w_lsq[:3],
        bilaplacian
        )
    
    from tictoc import tic, toc
    tic( 'build energy:' )
    
    ## Bi-Laplacian instead of Laplacian.
    if bilaplacian:
        L = gen_grid_laplacian2_with_boundary_reflection( rows, cols )
        L = L.T*L
    else:
        L = gen_symmetric_grid_laplacian( rows, cols )
    
    toc()
    tic( 'add constraints:' )
    
    system = L
    del L
    
    ## We must have at least one value constraint or our system is under-constrained.
    assert len( value_constraints_hard ) + len( value_constraints_soft ) > 0
    
    ## Least squares weight should be non-negative:
    assert w_lsq >= 0.
    
    tic( 'value constraints soft:' )
    if len( value_constraints_soft ) > 0:
        C, Crhs = gen_constraint_system( rows, cols, value_constraints = [ (i,j,0) for i,j in value_constraints_soft ] )
        
        #debugger()
        ## The system to solve:
        ## w_smooth * ( L * grid.ravel() ) + w_gradients * ( C.T * C * grid.ravel() ) = w_gradients * C.T * Crhs
        
        ## Is w_lsq iterable? If so, then we have a weight per soft constraint equation.
        if hasattr( w_lsq, '__iter__' ):
            w_lsq = asfarray( w_lsq )
            assert len( w_lsq ) == len( Crhs )
        else:
            w_lsq = w_lsq * ones( len( Crhs ) )
            
        from scipy import sparse
        W = sparse.coo_matrix( ( w_lsq, ( range( len( w_lsq ) ), range( len( w_lsq ) ) ) ) )
        
        CTW = C.T*W
        system = ( system + CTW*C )
    toc()
    
    ## NOTE: We must set hard constraints *after* we set soft constraints, because
    ##       the hard constraints set certain rows and columns to identity rows and columns.
    tic( 'value constraints hard (fast):' )
    if len( value_constraints_hard ) > 0:
        system = system_with_value_constraints3( system, value_constraints_hard, cols )
    toc()
    
    toc()
    
    import cvxopt, cvxopt.cholmod, cvxopt.umfpack
    tic( 'convert matrix for solving:' )
    system = system.tocoo()
    
    system = cvxopt.spmatrix( system.data, asarray( system.row, dtype = int ), asarray( system.col, dtype = int ) )
    
    toc()
    
    tic( 'facorize system of equations direct:' )
    factorized = cvxopt.umfpack.numeric( system, cvxopt.umfpack.symbolic( system ) )
    toc()
    
    def solve( hard_values = [], soft_values = [] ):
        
        ## We must have at least one value constraint or our system is under-constrained.
        assert len( value_constraints_hard ) == len( hard_values )
        assert len( value_constraints_soft ) == len( soft_values )
        ## All value constraint values should be vectors of length >= 1.
        assert all([ len( asarray( vals ).shape ) == 1 for vals in itertools.chain( hard_values, soft_values ) ])
        ## All value constraint values should be vectors with the same length.
        result_dim = list( set([ len( vals ) for vals in itertools.chain( hard_values, soft_values ) ]) )
        assert len( result_dim ) == 1
        result_dim = result_dim[0]
        
        rhs = zeros( ( rows*cols, result_dim ) )
        if len( value_constraints_soft ) > 0:
            rhs = ( rhs + CTW*gen_constraint_rhs( soft_values ) )
        if len( value_constraints_hard ) > 0:
            rhs = rhs_with_value_constraints3( rhs, value_constraints_hard, hard_values, cols )
        rhs = cvxopt.matrix( rhs )
        
        ## This function overwrites 'rhs' with the solution.
        cvxopt.umfpack.solve( system, factorized, rhs )
        
        result = array( rhs ).ravel().reshape( ( rows, cols, result_dim ) )
    
        ## We pass this test (with default epsilon with hard constraints, and within 1e-4 when w_lsq = 1e3):
        #assert allclose( result, solve_grid_linear_simple2( rows, cols, value_constraints_hard = [ (i,j,val) for (i,j), val in zip( value_constraints_hard, hard_values ) ], value_constraints_soft = [ (i,j,val) for (i,j), val in zip( value_constraints_soft, soft_values ) ], w_lsq = w_lsq, bilaplacian = bilaplacian ) )
        
        return result
    
    return solve

def solve_grid_linear_simple3( rows, cols, value_constraints_hard = [], value_constraints_soft = [], w_lsq = 1., bilaplacian = False ):
    '''
    Identical to solve_grid_linear_simple2(), except uses solve_grid_linear_simple3_solver() to create a solve() function that can be used repeatedly.
    '''
    solve = solve_grid_linear_simple3_solver( rows, cols, value_constraints_hard = [ (i,j) for i,j,val in value_constraints_hard ], value_constraints_soft = [ (i,j) for i,j,val in value_constraints_soft ], w_lsq = w_lsq, bilaplacian = bilaplacian )
    result = solve( hard_values = [ val for i,j,val in value_constraints_hard ], soft_values = [ val for i,j,val in value_constraints_soft ] )
    return result

def smooth_bumps( grid, bump_locations, bump_radius, **solve_grid_linear_kwargs ):
    '''
    Given a rows by cols 2D 'grid',
    a sequence 'bump_locations' of tuples ( i, j ) specifying the locations of bumps to smooth grid[ i,j ],
    a non-negative integer 'bump_radius' corresponding to the L1 radius of the bump in grid units (0 means just the 'bump_locations', 1 means up to 1 grid edge away, etc),
    and additional keyword arguments 'solve_grid_linear_kwargs'
    that will be passed to solve_grid_linear(),
    returns a new grid obtained by smoothing the regions of 'grid'
    given by 'bump_locations' and 'bump_radius'.
    
    NOTE: The argument 'w_lsq' for solve_grid_linear() has no effect here,
          because the smoothing is performed with hard constraints.
    
    tested:
    >>> g = zeros( ( 5, 5 ) )
    >>> gp = smooth_bumps( g, [ ( 3,3 ), ( 0,0 ), ( 0, 4 ) ], 0 )
    >>> gp = smooth_bumps( g, [ ( 3,3 ), ( 0,0 ), ( 0, 4 ) ], 2 )
    >>> ts = linspace( 0, pi, 5 )
    >>> g = 5*outer( sin( ts ), cos( ts ) )
    >>> gp = smooth_bumps( g, [ ( 3,3 ), ( 0,0 ), ( 0, 4 ) ], 1, bilaplacian = True )
    >>> import heightmesh
    >>> heightmesh.save_grid_as_OBJ( gp, 'smooth_bumps.obj' )
    >>> import trimesh_viewer
    >>> trimesh_viewer.view_mesh( heightmesh.grid2trimesh( gp, 'smooth_bumps' ) )
    '''
    
    if 'w_lsq' in solve_grid_linear_kwargs:
        print "WARNING: parameter 'w_lsq' has no effect smooth_bumps()"
    
    assert len( grid.shape ) == 2
    assert grid.shape[0] > 0
    assert grid.shape[1] > 0
    
    bump_locations = asarray( bump_locations, dtype = int )
    assert len( bump_locations.shape ) == 2 and bump_locations.shape[1] == 2
    ## Make sure bump_locations are in bounds.
    assert bump_locations[:,0].min() >= 0
    assert bump_locations[:,1].min() >= 0
    assert bump_locations[:,0].max() < grid.shape[0]
    assert bump_locations[:,1].max() < grid.shape[1]
    
    assert bump_radius >= 0
    assert int( bump_radius ) == bump_radius
    bump_radius = int( bump_radius )
    
    ### 1 Make a mask for the grid based on the bump locations expanded by
    ###   the bump radius.
    ### 2 Create value constraints for all locations not indicated by the mask.
    ### 3 Pass the result to solve_grid_linear().
    
    
    ### 1
    mask = zeros( grid.shape, dtype = bool )
    mask[ tuple( bump_locations.T ) ] = True
    #print mask
    
    from highlighting import grow
    mask = grow( mask, bump_radius )
    #print mask
    
    
    ### 2
    mask = logical_not( mask )
    value_constraints = [ ( i, j, grid[ i,j ] ) for i, j in zip( *where( mask ) ) ]
    #print value_constraints
    
    
    ### 3
    result = solve_grid_linear( grid.shape[0], grid.shape[1], value_constraints = value_constraints, **solve_grid_linear_kwargs )
    assert list( result.shape ) == list( grid.shape )
    
    
    return result

def smooth_bumps2( grid, bump_locations, bump_radius, smooth_bumps_iterations, smooth_func = None, **solve_grid_linear_kwargs ):
    '''
    Given a rows by cols 2D 'grid',
    a sequence 'bump_locations' of tuples ( i, j ) specifying the locations of bumps to smooth grid[ i,j ],
    a non-negative integer 'bump_radius' corresponding to the L1 radius of the bump in grid units (0 means just the 'bump_locations' and has no effect, 1 means up to 1 grid edge away, etc),
    a non-negative integer 'smooth_bumps_iterations' corresponding to the number of smoothing iterations,
    optional argument 'smooth_func' which can be either 'median' or 'average',
    and additional keyword arguments 'solve_grid_linear_kwargs'
    that will be passed to solve_grid_linear(),
    returns a new grid obtained by solving for a new 'grid' with
    successively smoother constraints at the 'bump_locations';
    this smoothing process will take place 'smooth_bumps_iterations' times
    and the smoother constraints will be obtained by taking the average
    or median gradient of an area 'bump_radius' around each location
    in 'bump_locations'.
    
    tested:
    >>> g = zeros( ( 5, 5 ) )
    >>> gp = smooth_bumps2( g, [ ( 3,3 ), ( 0,0 ), ( 0, 3 ) ], 0, 0 )
    >>> gp = smooth_bumps2( g, [ ( 3,3 ), ( 0,0 ), ( 0, 3 ) ], 0, 1 )
    >>> gp = smooth_bumps2( g, [ ( 3,3 ), ( 0,0 ), ( 0, 3 ) ], 0, 2 )
    >>> gp = smooth_bumps2( g, [ ( 3,3 ), ( 0,0 ), ( 0, 3 ) ], 2, 1 )
    >>> ts = linspace( 0, pi, 5 )
    >>> g = 5*outer( sin( ts ), cos( ts ) )
    >>> gp = smooth_bumps2( g, [ ( 3,3 ), ( 0,0 ), ( 0, 3 ) ], 1, 1, bilaplacian = True )
    >>> import heightmesh
    >>> heightmesh.save_grid_as_OBJ( gp, 'smooth_bumps.obj' )
    >>> import trimesh_viewer
    >>> trimesh_viewer.view_mesh( heightmesh.grid2trimesh( gp, 'smooth_bumps' ) )
    '''
    
    assert len( grid.shape ) == 2
    assert grid.shape[0] > 0
    assert grid.shape[1] > 0
    
    bump_locations = asarray( bump_locations, dtype = int )
    assert len( bump_locations.shape ) == 2 and bump_locations.shape[1] == 2
    ## Make sure bump_locations are in bounds.
    assert bump_locations[:,0].min() >= 0
    assert bump_locations[:,1].min() >= 0
    ## UPDATE: Make sure bump_locations are not on the last row or column,
    ##         since they are derivative constraints.
    ## TODO: Relax this restriction by factoring the code in
    ##       convert_grid_normal_constraints_to_dx_and_dy_constraints()
    ##       that jitters constraints in the last row or column.
    assert bump_locations[:,0].max() < grid.shape[0]-1
    assert bump_locations[:,1].max() < grid.shape[1]-1
    
    assert bump_radius >= 0
    assert int( bump_radius ) == bump_radius
    bump_radius = int( bump_radius )
    
    assert smooth_bumps_iterations >= 0
    assert int( smooth_bumps_iterations ) == smooth_bumps_iterations
    smooth_bumps_iterations = int( smooth_bumps_iterations )
    
    if smooth_func is None: smooth_func = 'median'
    assert smooth_func in ( 'median', 'average' )
    
    ### 1 For each location in 'bump_locations', collect the neighbor
    ###   indices for use in determining smoother and smoother constraints.
    ### 2 In a loop:
    ### 2a Take the median dx and dy in the area around each location
    ###    in 'bump_locations'.
    ### 2b Solve for a new grid.
    
    
    ## If we want 0 smoothing iterations, we are already done.
    ## Return a copy (as per the documentation).
    if smooth_bumps_iterations == 0:
        return array( grid )
    
    #debugger()
    
    ### 1
    bump_neighbors = []
    for loc in bump_locations:
        ## The mask is without the right-most row and column, since
        ## each location specifies a gradient in the +row and +column
        ## directions.
        mask = zeros( grid[:-1,:-1].shape, dtype = bool )
        mask[ tuple( loc ) ] = True
        #print mask
        
        from highlighting import grow
        mask = grow( mask, bump_radius )
        #print mask
        
        bump_neighbors.append( where( mask ) )
    
    #print 'bump_neighbors:', bump_neighbors
    
    
    ### 2
    smooth_func = { 'median': median, 'average': average }[ smooth_func ]
    for i in xrange( smooth_bumps_iterations ):
        
        ### 2a
        dx_constraints = []
        dy_constraints = []
        for loc, indices in zip( bump_locations, bump_neighbors ):
            dx_constraints.append( ( loc[0], loc[1], smooth_func( ( grid[1:,:] - grid[:-1,:] )[ indices ] ) ) )
            dy_constraints.append( ( loc[0], loc[1], smooth_func( ( grid[:,1:] - grid[:,:-1] )[ indices ] ) ) )
        
        #print 'dx_constraints:', dx_constraints
        #print 'dy_constraints:', dy_constraints
        
        ### 2b
        grid = solve_grid_linear( grid.shape[0], grid.shape[1], dx_constraints = dx_constraints, dy_constraints = dy_constraints, **solve_grid_linear_kwargs )
    
    return grid

def convert_grid_normal_constraints_to_dx_and_dy_constraints( rows, cols, normal_constraints ):
    '''
    Given dimensions of a 'rows' by 'cols' 2D grid and
    a sequence 'normal_constraints' of tuples ( i, j, 3D normal ),
    returns a 2-tuple ( dx_constraints, dy_constraints ), where
        'dx_constraints' is a sequence of tuples ( i, j, dx ) specifying the value of grid[ i+1,j ] - grid[ i,j ],
        'dy_constraints' is a sequence of tuples ( i, j, dy ) specifying the value of grid[ i,j+1 ] - grid[ i,j ],
    suitable for passing to solve_grid_linear().
    
    NOTE: The 3D coordinate space of the normal is ( x = +i, y = +j, z = direction obtained by cross( +i, +j ) ).
          This is a right-hand coordinate system.
    
    NOTE: To be valid normals on a height map, all normals must have z > 0.
    
    NOTE: solve_grid_linear() will be under-constrained if no 'value_constraints' are passed to it.
          To prevent such a scenario, simply add a single, arbitrary constraint, such as value_constraints = [(0,0,0)].
    
    used
    '''
    
    ## Our task is impossible if the grid is 1x1.
    assert rows > 0 and cols > 0
    
    ## If there are constraints along the right or bottom edge, shift them over
    ## by one so long as this won't push any constraints off the left edge.
    ## If this works to handle constraints along the far edges, it also
    ## preserves the distance between constraints, so it's better than
    ## shifting constraints locally (our fallback solution below).
    all_is = set([ i for i,j,normal in normal_constraints ])
    all_js = set([ j for i,j,normal in normal_constraints ])
    ## Shift everything over by one if we have a constraint on the far edge
    ## but not the near.
    if (rows-1) in all_is and not 0 in all_is: normal_constraints = [ (i-1,j,normal) for i,j,normal in normal_constraints ]
    if (cols-1) in all_js and not 0 in all_js: normal_constraints = [ (i,j-1,normal) for i,j,normal in normal_constraints ]
    
    ## The above will not have worked if there are constraints on both the
    ## far and near edges.  As a fallback, shift those constraints locally
    ## (if there isn't a constraint on the immediate near-side).
    ## First collect safe constraints.
    #ij2normal = dict( [ ((i,j), normal) for i,j,normal in normal_constraints if i < rows-1 and j < cols-1 ] )
    #for i,j,normal in normal_constraints:
    ij2normal = {}
    unsafe = []
    for i,j,normal in normal_constraints:
        if i < rows-1 and j < cols-1:
            ij2normal[ (i,j) ] = normal
        else:
            unsafe.append( ( i, j, normal ) )
    ## Then shift unsafe constraints and add them, if possible.
    for i,j,normal in unsafe:
        if i == rows-1 and j == cols-1:
            if (i-1,j-1) not in ij2normal:
                ij2normal[ (i-1,j-1) ] = normal
        elif i == rows-1:
            if (i-1,j) not in ij2normal:
                ij2normal[ (i-1,j) ] = normal
        elif j == cols-1:
            if (i,j-1) not in ij2normal:
                ij2normal[ (i,j-1) ] = normal
        else:
            raise RuntimeError, "We are only iterating over unsafe indices!"
    
    dx_constraints = []
    dy_constraints = []
    for ( i, j ), normal in ij2normal.iteritems():
        ## We will create dx/dy constraints by pretending that there is
        ## a triangle with vertices (i,j), (i+1,j), (i,j+1) whose normal is 'normal'.
        ## The dx constraint will be a constraint on the edge (i,j), (i+1,j),
        ## and the dy constraint will be a constraint on the edge (i,j), (i,j+1).
        
        ## Viewing the triangle's (i,j), (i+1,j) edge in the xz plane,
        ## the normal has a slope (rise/run) of normal.z / normal.x.
        ## The dx constraint is the slope of the surface's z value
        ## in the x direction.  This slope should be perpendicular to
        ## normal.z / normal.x,
        ## therefore dx = -normal.x / normal.z.
        ## Similarly for the dy constraint.
        n = normal
        ## TODO Q: Should I allow the height map to have discontinuities at
        ##         silhouettes, like a normal with a z = 0?
        #assert n[2] > 0
        if n[2] < 1e-5:
            print 'Silhouette normal!  TODO: Make it a height map discontinuity.  Skipping for now...'
            continue
        
        dx_constraints.append( ( i,j, -n[0]/n[2] ) )
        dy_constraints.append( ( i,j, -n[1]/n[2] ) )
    
    return dx_constraints, dy_constraints

def test_grid_poisson():
    #result = solve_grid_linear( 3, 3, [(1,0,1.),(1,1,1.),(1,2,1.)], [(0,1,0.),(1,1,0.),(2,1,0.)] )
    
    print 'laplacian:'
    result = solve_grid_linear( 4, 3, [(1,0,1.),(1,1,1.),(1,2,1.)], [] )
    print result.round(2)
    print 'bilaplacian:'
    result = solve_grid_linear( 4, 3, [(1,0,1.),(1,1,1.),(1,2,1.)], [], bilaplacian = True )
    print result.round(2)
    
    #result = solve_grid_linear( 10, 10, [(1,0,1.),(1,1,1.),(1,2,1.)], [] ) #[(0,1,0.),(1,1,0.),(2,1,0.)] )
    #result = solve_grid_linear( 10, 10, [], [] )
    #print result.round(2)
    
    #debugger()
    #pass

def solve_linear( mesh, normal_constraints ):
    '''
    Given a TriMesh 'mesh',
    a sequence 'normal_constraints' of tuples ( face index into mesh.faces[], 3D normal ),
    returns the solution to the thin plate energy (NOT the Poisson Equation)
    on the domain given by 'mesh' with normals given by 'normal_constraints'.
    '''
    
    from scipy import sparse
    import scipy.sparse.linalg
    
    ## Normal constraints must be normalized.
    assert False not in [ abs( dot( n, n ) - 1. ) <= 1e-5 for fi, n in normal_constraints ]
    ## Normal constraints must be on valid face indices.
    assert False not in [ fi >= 0 and fi < len( mesh.faces ) for fi, n in normal_constraints ]
    
    output = mesh.copy()
    output.vs = asarray( output.vs )
    
    ### Smoothness term (system).
    import laplacian_system
    output.vs[:,2] = 0.
    Acot, areas_cot = laplacian_system.gen_cotangent_laplacian_matrix( output, None )
    M = sparse.identity( len( output.vs ) )
    M.setdiag( areas_cot )
    Acot = Acot.tocsr()
    Acot = M.tocsr() * Acot * Acot
    #face_areas = array( output.face_areas )
    
    
    ### Constrain normals (system).
    ## The following three lists define linear equations in the z coordinates: \sum_i \sum_j zs[ indices[i][j] ] * vals[i][j] = rhs[i]
    indices = []
    vals = []
    rhs = []
    ## Do this by constraining face edges according to the undeformed face rotating to have the desired normal.
    ## In the Shading-Based Surface Editing work, we tried constraining the dot product and saw a lot of area shrinkage.
    ## On the other hand, since only z is free, perhaps this isn't an issue.
    ## Find the average normal inside a face, and constrain that face to that normal.  Weight not by the face's area, but by the number or confidence of samples.
    for ofi, normal in normal_constraints:
        #area = output.face_areas[ ofi ]
        #area = mesh.face_areas[ mfi ]
        #area = face_areas[ ofi ]
        
        ## Something like this would work if the entire edge were free, and not just the z coordinate.
        ## (In the Shading-Based Surface Editing project, I found that constraining the face to a rotated face worked better than constraining the dot product to 0;
        ## all three coordinates, xyz, of each vertex were free and constraining the dot product to 0 would shrink faces.)
        '''
        axis, angle = get_axis_angle_rotation( output.face_normals[ ofi ], normal )
        e01 = rotate( output.vs[ output.faces[ ofi ][1] ] - output.vs[ output.faces[ ofi ][0] ], axis, angle )
        e02 = rotate( output.vs[ output.faces[ ofi ][2] ] - output.vs[ output.faces[ ofi ][0] ], axis, angle )
        
        indices.append( [ output.faces[ ofi ][1], output.faces[ ofi ][0] ] )
        vals.append( [ 1., -1. ] )
        rhs.append( e01 )
        
        indices.append( [ output.faces[ ofi ][2], output.faces[ ofi ][0] ] )
        vals.append( [ 1., -1. ] )
        rhs.append( e02 )
        '''
        
        ## Instead only z is free, so we can only place constraints on the z values; the desired z value is the one such that the dot product is 0.
        ## You get exactly the same constraint if you constrain the gradient along the triangle edge like so:
        ##    (f[xy_i] - f[xy_j])/|xy_i - xy_j| = z(xy_i) - z(xy_j)/|xy_i - xy_j|
        ## where z( xy ) is a function mapping the 2D point xy to the z component of its projection onto the plane defined by the normal.
        n = normal
        
        ## Add some random noise to test.
        #n += .2 * random.rand(3)
        #n /= sqrt( dot( n,n ) )
        
        ## dot( edge, n ) = 0 = dot( edge.xy, n.xy ) + edge.z * n.z = 0 <=> edge.z = -dot( edge.xy, n.xy ) / n.z
        e01 = output.vs[ output.faces[ ofi ][1] ] - output.vs[ output.faces[ ofi ][0] ]
        e02 = output.vs[ output.faces[ ofi ][2] ] - output.vs[ output.faces[ ofi ][0] ]
        
        indices.append( [ output.faces[ ofi ][1], output.faces[ ofi ][0] ] )
        vals.append( [ 1., -1. ] )
        rhs.append( -dot( e01[:2], n[:2] ) / n[2] )
        
        indices.append( [ output.faces[ ofi ][2], output.faces[ ofi ][0] ] )
        vals.append( [ 1., -1. ] )
        rhs.append( -dot( e02[:2], n[:2] ) / n[2] )
    
    ## We are also underdetermined if we don't constrain an additional point.
    ## Constrain zs[0] = 0.
    indices.append( [ 0 ] )
    vals.append( [ 1. ] )
    rhs.append( 0. )
    
    ## Build the constraints system.
    rows = [ [row] * len( cols ) for row, cols in enumerate( indices ) ]
    #debugger()
    C = sparse.coo_matrix( ( concatenate( vals ), ( concatenate( rows ), concatenate( indices ) ) ), shape = ( len( rows ), len( output.vs ) ) )
    Crhs = asarray( rhs )
    
    
    ## Smoothness term.
    w_smooth = 1.
    ## Constraints (normals and z0 = 0) term.
    w_normal = 1e5
    
    ## The system:
    ## w_smooth * ( Acot * zs ) + w_normal * ( C.T * C * zs ) = w_normal * C.T * Crhs
    zs = sparse.linalg.spsolve( w_smooth * Acot + ( w_normal * C.T ) * C, ( w_normal * C.T ) * Crhs )
    
    output.vs[:,2] = zs
    output.positions_changed()
    
    return output

def solve_nonlinear( mesh, normal_constraints ):
    '''
    Given a TriMesh 'mesh',
    a sequence 'normal_constraints' of tuples ( face index into mesh.faces[], 3D normal ),
    returns the solution to the thin plate energy (NOT the Poisson Equation)
    on the domain given by 'mesh' with normals given by 'normal_constraints'.
    
    NOTE: doesn't work
    '''
    
    ## Normal constraints must be normalized.
    assert False not in [ abs( dot( n, n ) - 1. ) <= 1e-5 for fi, n in normal_constraints ]
    ## Normal constraints must be on valid face indices.
    assert False not in [ fi >= 0 and fi < len( mesh.faces ) for fi, n in normal_constraints ]
    
    output = mesh.copy()
    
    import tnc
    rc, nfeval, zs = tnc.minimize(
        get_E_and_grad_nonlinear( output, normal_constraints ),
        list( zeros( len( output.vs ) ) )
        )
    output.vs[:,2] = zs
    output.positions_changed()
    
    return output

def get_E_and_grad_nonlinear( mesh, normal_constraints ):
    import laplacian_editing
    
    def E( zs ):
        mesh.vs[:,2] = zs
        mesh.positions_changed()
        
        e = 0.
        
        ## Smoothness term.
        w_smooth = 1.
        for vi in xrange( len( mesh.vs ) ):
            
            one_ring_indices = mesh.vertex_vertex_neighbors( vi )
            one_ring = [ mesh.vs[ vn ] for vn in one_ring_indices ]
            is_boundary = mesh.vertex_is_boundary( vi )
            
            cot_alpha, cot_beta, A = laplacian_editing.cotangentWeights( mesh.vs[ vi ], one_ring, is_boundary )
            indices, values = laplacian_editing.Hn_coefficients(
                vi,
                one_ring_indices,
                is_boundary,
                cot_alpha, cot_beta, A
                )
            
            Hn = ( mesh.vs[ indices, : ] * values[:,newaxis] ).sum( 0 )
            e += w_smooth * dot( Hn, Hn ) / A
        
        
        ## Constrain normals.
        w_normal = 1e5
        for fi, normal in normal_constraints:
            area = mesh.face_areas[ fi ]
            dp = dot( mesh.face_normals[ fi ], normal )
            e += w_normal * area * acos( dp )**2
        
        
        ## We are also underdetermined if we don't constrain an additional point.
        ## Constrain zs[0] = 0.
        w_point = 1.
        e += w_point * (zs[0])**2
        
        return e
    
    Nits = [0]
    def E_and_grad( zs ):
        Nits[0] += 1
        if Nits[0] >= 10: return None
        
        e = E( zs )
        
        ## Finite differencing
        h = 1e-7
        grad = zeros( len( zs ) )
        for zi in xrange( len( zs ) ):
            old_zi = zs[zi]
            zs[zi] += h
            
            grad[ zi ] = E( zs )
            
            zs[zi] = old_zi
        
        grad -= e
        grad *= (1/h)
        
        print 'e:', e
        #print 'grad:', grad
        
        return e, list( grad )
    
    return E_and_grad

def shape_operator_for_2D_points_and_normals( points, normals ):
    '''
    Given a sequence of 2d 'points' and corresponding 3d unit 'normals'
    where the center point is assumed to be the first element in the sequence,
    returns the 3-by-2 matrix mapping a 2D offset from the center point to
    a 3D offset for the normal of the center point;
    in other words, the normal at a 2D point p_i near the center point p_0 can be found by
    multiplying the matrix * ( p_i - p_0 ) and adding the result to the normal at p_0.
    This matrix is -1 times the shape operator, except that it maps from 2D to 3D.
    '''
    
    ## We want to find the matrix M that minimizes the squared error for the given input data.
    ## We will express this as:
    ## E[M] = \sum_i w_i | M ( p_i - p_0 ) - ( n_i - n_0 ) |^2
    ## where p_i, n_i are a 2D point and 3D normal pair,
    ## p_0, n_0 are the center point and normal,
    ## and w_i is the weight associated with the given point and normal.
    ## Then dE[M] / dM = \sum_i w_i 2 ( M ( p_i - p_0 ) - ( n_i - n_0 ) ) ( p_i - p_0 )^T
    ## We want dE[M]/dM = 0, so
    ## 0 = dE[M]/dM = \sum_i w_i ( M ( p_i - p_0 ) - ( n_i - n_0 ) ) ( p_i - p_0 )^T
    ##    = \sum_i w_i M ( p_i - p_0 ) ( p_i - p_0 )^T - \sum_i w_i ( n_i - n_0 ) ( p_i - p_0 )^T
    ##    =  M \sum_i w_i ( p_i - p_0 ) ( p_i - p_0 )^T - \sum_i w_i ( n_i - n_0 ) ( p_i - p_0 )^T
    ## <=>
    ## M \sum_i w_i ( p_i - p_0 ) ( p_i - p_0 )^T = \sum_i w_i ( n_i - n_0 ) ( p_i - p_0 )^T
    ## <=>
    ## M = \sum_i w_i ( n_i - n_0 ) ( p_i - p_0 )^T ( \sum_i w_i ( p_i - p_0 ) ( p_i - p_0 )^T )^{-1}
    
    ## Let's use w_i = 1/distance( p_i, p_0 ).
    points = asarray( points, dtype = float )
    normals = asarray( normals, dtype = float )
    dp = points[1:] - points[0]
    dn = normals[1:] - normals[0]
    w = 1. / sqrt( ( dp**2 ).sum(-1) )
    print 'w:', w
    
    RHS = asarray( [ outer( wi * ni, pi ) for ( wi, ni, pi ) in zip( w, dn, dp ) ] ).sum(0)
    LHS = asarray( [ outer( wi * pi, pi ) for ( wi, pi ) in zip( w, dp ) ] ).sum(0)
    from numpy import linalg
    M = dot( RHS, linalg.inv( LHS ) )
    return M

def test_shape_operator():
    ps = [ (0,0), (1,0), (0,1) ]
    ns = [ (0,0,1), (1,0,0), (0,1,0) ]
    M = shape_operator_for_2D_points_and_normals( ps, ns )
    print 'ps:', ps
    print 'ns:', ns
    print M
    ps = asarray( ps )
    ns = asarray( ns )
    Mns = ns[0] + dot( M, ( ps[1:] - ps[0] ).T ).T
    print 'ns[0] + dot( M, ps[1:] - ps[0] ):', Mns
    print 'ns[1:] - Mns.T:', ns[1:] - Mns
    u, s, vh = linalg.svd( M )
    print 'u:', u
    print 's:', s
    print 'vh:', vh
    #debugger()
    #pass

def normalize_image( img ):
    '''
    Given a 2D array, linearly maps the smallest value to 0 and the largest value to 1.
    '''
    
    ## Convert to a floating point array because of the division.
    if img.dtype.kind != 'f':
        img = asarray( img, dtype = float )
    
    img = img.squeeze()
    ## For greyscale, shape should be 2, but for color it could be 3.
    #assert len( img.shape ) == 2
    assert len( img.shape ) in (2,3)
    
    min_val = img.min()
    max_val = img.max()
    result = ( img - min_val ) / ( max_val - min_val )
    return result

def float_img2char_img( img ):
    '''
    Given a 2D array representing an image with values from 0 to 1, returns a uint8 image ranging from 0 to 255.
    '''
    
    ## Specifying dtype=uint8 is crucial, otherwise the png came out as garbage.
    result = asarray( ( 255.*img ).clip( 0, 255 ).round(), dtype = uint8 )
    return result

def normalize_to_char_img( img ):
    '''
    Given a 2D array representing an image, returns a uint8 array representing
    the image after linearly mapping the values to lie within the
    range [0,255].
    '''
    return float_img2char_img( normalize_image( img ) )

def test_cut_edges():
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
    
    mask = zeros( ( rows, cols ), dtype = bool )
    mask[br0:br1,bc0:bc1] = True
    
    cut_edges = cut_edges_from_mask( mask )
    
    ## Make the center a raised bump
    dxs = []
    dxs.extend([ ( br0, col, 1. ) for col in xrange( bc0, bc1 ) ])
    dxs.extend([ ( br1-2, col, -1. ) for col in xrange( bc0, bc1 ) ])
    
    dys = []
    dys.extend([ ( row, bc0, 1. ) for row in xrange( br0, br1 ) ])
    dys.extend([ ( row, bc1-2, -1. ) for row in xrange( br0, br1 ) ])
    
    import Image
    import heightmesh
    
    sol_tilted = solve_grid_linear( rows, cols, [ ( 0, 0, 1 ) ], [ ( 0, 0, 1 ) ], [ ( 0, 0, 0. ), ( 1, 1, 2. ) ], bilaplacian = True )
    Image.fromarray( normalize_to_char_img( sol_tilted ) ).save( 'sol_tilted.png' )
    print '[Saved "sol_tilted.png".]'
    heightmesh.save_grid_as_OBJ( sol_tilted, 'sol_tilted.obj' )
    
    sol_nomask = solve_grid_linear( rows, cols, dxs, dys, [ ( 0, 0, 0. ), ( rows-1, 0, 0. ), ( 0, cols-1, 0. ), ( rows-1, cols-1, 0. ), ], bilaplacian = True )
    Image.fromarray( normalize_to_char_img( sol_nomask ) ).save( 'sol_nomask.png' )
    print '[Saved "sol_nomask.png".]'
    heightmesh.save_grid_as_OBJ( sol_nomask, 'sol_nomask.obj' )
    
    sol_mask = solve_grid_linear( rows, cols, dxs, dys, [
        ( 0, 0, 0. ), ( rows-1, 0, 0. ), ( 0, cols-1, 0. ), ( rows-1, cols-1, 0. ),
        ( br0, bc0, br0 ), #( br1-1, bc0, br0 ), ( br0, bc1-1, br0 ), ( br1-1, bc1-1, br0 )
        ], bilaplacian = True, cut_edges = cut_edges )
    Image.fromarray( normalize_to_char_img( sol_mask ) ).save( 'sol_mask.png' )
    print '[Saved "sol_mask.png".]'
    heightmesh.save_grid_as_OBJ( sol_mask, 'sol_mask.obj' )
    
    sol_mask_tilted = solve_grid_linear( rows, cols, dxs + [ ( 0, 0, 1 ) ], dys + [ ( 0, 0, 1 ) ],
        [
            ( 0, 0, 0. ), ( 1, 1, 2. ),
            ( br0, bc0, br0 ), #( br1-1, bc0, br0 ), ( br0, bc1-1, br0 ), ( br1-1, bc1-1, br0 )
            #( 0, 0, 0. ), ( br0, bc0, 1. )
        ], bilaplacian = True, cut_edges = cut_edges )
    Image.fromarray( normalize_to_char_img( sol_mask_tilted ) ).save( 'sol_mask_tilted.png' )
    print '[Saved "sol_mask_tilted.png".]'
    heightmesh.save_grid_as_OBJ( sol_mask_tilted, 'sol_mask_tilted.obj' )
    
    connected_dx, connected_dy = zero_dx_dy_constraints_from_cut_edges( cut_edges )
    sol_mask_tilted_connected = solve_grid_linear(
        rows, cols, dxs + [ ( 0, 0, 1 ) ] + connected_dx, dys + [ ( 0, 0, 1 ) ] + connected_dy,
        [
            ( 0, 0, 0. ), #( rows-1, 0, 0. ), ( 0, cols-1, 0. ), ( rows-1, cols-1, 0. ),
            #( br0, bc0, 1. ), ( br1, bc0, 1. ), ( br0, bc1, 1. ), ( br1, bc1, 1. )
            #( 0, 0, 0. ), ( br0, bc0, 1. )
        ], bilaplacian = True, cut_edges = cut_edges )
    Image.fromarray( normalize_to_char_img( sol_mask_tilted_connected ) ).save( 'sol_mask_tilted_connected.png' )
    print '[Saved "sol_mask_tilted_connected.png".]'
    heightmesh.save_grid_as_OBJ( sol_mask_tilted_connected, 'sol_mask_tilted_connected.obj' )

def test_solve_grid_linear_simpleN( solve_grid_linear_simpleN ):
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
    
    import Image
    import heightmesh
    
    ## This works with K = 1 or K = 3.
    K = 1
    value_constraints = [
        ( 0, 0, [0.]*K ), ( rows-1, 0, [0.]*K ), ( 0, cols-1, [0.]*K ), ( rows-1, cols-1, [0.]*K ),
        ( br0, bc0, [br0]*K ), ( br1-1, bc0, [br0]*K ), ( br0, bc1-1, [br0]*K ), ( br1-1, bc1-1, [br0]*K )
        ]
    
    sol_hard = solve_grid_linear_simpleN( rows, cols, value_constraints_hard = value_constraints, bilaplacian = True )
    sol_soft = solve_grid_linear_simpleN( rows, cols, value_constraints_soft = value_constraints, w_lsq = 1e3, bilaplacian = True )
    sol_soft2 = solve_grid_linear_simpleN( rows, cols, value_constraints_soft = value_constraints, w_lsq = [1e3]*len(value_constraints), bilaplacian = True )
    #print 'maximum absolute difference for sol_soft when using a per-equation constraint:', abs( sol_soft2 - sol_soft ).max()
    assert abs( sol_soft2 - sol_soft ).max() < 1e-7
    
    sol_mixed = solve_grid_linear_simpleN( rows, cols, value_constraints_hard = value_constraints, value_constraints_soft = [ ( (br0+br1)//2, (bc0+bc1)//2, [.5*br0]*K ) ], w_lsq = 1e3, bilaplacian = True )
    sol_mixed2 = solve_grid_linear_simpleN( rows, cols, value_constraints_hard = value_constraints, value_constraints_soft = [ ( (br0+br1)//2, (bc0+bc1)//2, [.5*br0]*K ) ], w_lsq = [1e3]*1, bilaplacian = True )
    #print 'maximum absolute difference for sol_mixed when using a per-equation constraint:', abs( sol_mixed2 - sol_mixed ).max()
    assert abs( sol_mixed2 - sol_mixed ).max() < 1e-7
    
    Image.fromarray( normalize_to_char_img( sol_hard ) ).save( 'sol_hard.png' )
    Image.fromarray( normalize_to_char_img( sol_hard ) ).save( 'sol_soft.png' )
    Image.fromarray( normalize_to_char_img( sol_mixed ) ).save( 'sol_mixed.png' )
    print '[Saved "sol_soft.png".]'
    print '[Saved "sol_hard.png".]'
    print '[Saved "sol_mixed.png".]'
    #heightmesh.save_grid_as_OBJ( sol_nomask, 'sol_hard.obj' )

def debug_matrix_rank( M ):
    print linalg.svd( M.todense() )[1] > 1e-10
def view_grid( grid ):
    '''
    A helper function for viewing the grid.
    
    Prints an error rather than dying if the
    necessary helper modules are missing.
    '''
    assert len( grid.shape ) == 2
    
    try:
        import heightmesh, trimesh_viewer
        trimesh_viewer.view_mesh( heightmesh.grid2trimesh( grid ), 'grid' )
    except ImportError:
        print "Can't use view_grid() because some modules are missing."

def main():
    import sys
    from trimesh import TriMesh
    
    #test_solve_grid_linear_simpleN( solve_grid_linear_simple2 )
    test_solve_grid_linear_simpleN( solve_grid_linear_simple3 )
    sys.exit(0)
    
    #test_cut_edges()
    #sys.exit(0)
    
    #test_grid_poisson()
    #sys.exit(0)
    
    ## Q: Why isn't this a tilted plane?
    ## A1: I have to fix the entire boundary (two rows for bilaplacian, one row for laplacian).
    ## A2: There is no way a plane can be recreated at a boundary vertex (edge or corner)
    ##     without looking at the 2-ring (consider a line or a plane whose
    ##     gradient is diagonal to the grid).
    # solve_grid_linear( 9, 9, [ ( 0,0,1 ) ], [ ( 0,0,1 ) ], [ ( 0, 0, 0. ) ], bilaplacian = True )
    ## This should not be a tilted plane, it should return to dx = dy = 0 away from the corner:
    # solve_grid_linear( 9, 9, [ ( 0,0,1 ) ], [ ( 0,0,1 ) ], [ ( 0, 0, 0. ) ], bilaplacian = False )
    ## A3: It is a tilted plane if I use gen_grid_laplacian2_with_boundary_reflection(),
    ##     but I need to do something about the corner vertices...
    ## UPDATE: Maybe not.  They are involved in equations in the system.
    ##         Their columns aren't zero even if their rows are.
    ##         This means that the rank is lower, so we need more constraints,
    ##         but it's not clear to me what properties these constraints must
    ##         have.
    # solve_grid_linear( 9, 9, [ ( 4,4,1 ) ], [ ( 4,4,1 ) ], [ ( 4, 4, 0. ) ], bilaplacian = True )
    '''
from recovery import *
view_grid( solve_grid_linear( 9, 9, dx_constraints = [ ( 4,4,1 ) ], dy_constraints = [ ( 4,4,1 ) ], value_constraints = [ ( 4, 4, 0. ), ( 0,0, 0. ), ( 0, 8, 0. ), ( 8, 0, 0. ), ( 8, 8, 0. ) ], bilaplacian = True ) )
g = solve_grid_linear( 9, 9, dx_constraints = [ ( 4,4,1 ) ], dy_constraints = [ ( 4,4,1 ) ], value_constraints = [ ( 2, 2, 0. ), ], bilaplacian = True )
g = solve_grid_linear( 9, 9, dx_constraints = [ ( 0,0,1 ) ], dy_constraints = [ ( 0,0,1 ) ], value_constraints = [ ( 4, 4, 0. ) ], bilaplacian = True )
g = solve_grid_linear( 9, 9, [ ( 0,0,1 ) ], [ ( 0,0,1 ) ], [ ( 0, 0, 0. ) ], bilaplacian = True )
view_grid( g )
'''
    
    ## Q: Why isn't this a slanted line?
    ## A1: I have to fix the sides (one value on either side for laplacian, two values for bilaplacian).
    # L = gen_symmetric_grid_laplacian( 9, 1 )
    # ( L.T * L ) * linspace( 0, 8, 9 )
    ## These are zero:
    # gen_symmetric_grid_laplacian( 9, 1 ) * ones( 9 )
    # gen_symmetric_grid_laplacian( 9, 1 ) * zeros( 9 )
    ## This is not:
    # gen_symmetric_grid_laplacian( 9, 1 ) * linspace( 0, 8, 9 )
    ## A2: It is a slanted line if I use gen_grid_laplacian2_with_boundary_reflection().
    # gen_grid_laplacian2_with_boundary_reflection( 9, 1 ) * linspace( 0, 8, 9 )
    ## Simple grids for testing:
    # L = gen_grid_laplacian2_with_boundary_reflection( 9, 1 )
    # L = gen_grid_laplacian2_with_boundary_reflection( 9, 2 )
    
    ## These are straight lines:
    # solve_grid_linear( 9, 1, value_constraints = [ ( 0,0,0 ), (1,0,1), (7,0,7), ( 8,0,8 ) ], bilaplacian = True )
    # solve_grid_linear( 9, 1, value_constraints = [ ( 0,0,0 ), (1,0,1), (7,0,7), ( 8,0,8 ) ], bilaplacian = False )
    # solve_grid_linear( 9, 1, value_constraints = [ ( 0,0,0 ), ( 8,0,8 ) ], bilaplacian = False )
    ## These are not:
    ## UPDATE: Using gen_grid_laplacian2_with_boundary_reflection(), if bilaplacian = True,
    ##         these are straight lines!
    # solve_grid_linear( 9, 1, value_constraints = [ ( 0,0,0 ), ( 8,0,8 ) ], bilaplacian = True )
    # solve_grid_linear( 9, 1, value_constraints = [ ( 0,0,0 ), (1,0,1) ], bilaplacian = True )
    # solve_grid_linear( 9, 1, value_constraints = [ ( 0,0,0 ), (1,0,1) ], bilaplacian = False )
    
    
    #test_shape_operator()
    
    recovery1( TriMesh.FromOBJ_FileName( sys.argv[1] ) )
    ## e.g.
    # /usr/bin/python2.5 recovery.py results/output2.obj
    ## and then check the results for a linear solve
    # ./compareobj.py output.obj results/output3.obj 0.
    ## or the results of a non-linear solve
    # ./compareobj.py output.obj results/output3-nonlinear.obj 0.

if __name__ == '__main__': main()
