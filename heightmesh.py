#!/usr/bin/env python

from numpy import *
from itertools import izip as zip

try:
    from pydb import debugger
    
    ## Also add an exception hook.
    import pydb, sys
    sys.excepthook = pydb.exception_hook
    
except ImportError:
    def debugger():
        print '==== Eek, no debugger()! ===='

def save_grid_as_OBJ( grid, objpath, four_eight = False, mask = None ):
    mesh = grid2trimesh( grid, four_eight = four_eight, mask = mask )
    print '[Generated %s triangle mesh.]' % ( 'four-eight' if four_eight else 'regular', )
    mesh.write_OBJ( objpath )

def grid2trimesh( grid, four_eight = False, mask = None ):
    vs, faces, uvs = grid2vertices_and_faces_and_uvs( grid, four_eight = four_eight )
    
    from trimesh import TriMesh
    
    mesh = TriMesh()
    mesh.vs = vs
    mesh.faces = faces
    if uvs is not None:
        assert len( uvs ) == len( vs )
        mesh.uvs = uvs
    
    if mask is not None:
        mask = asarray( mask, dtype = bool )
        assert mask.shape == grid.shape
        
        #'''
        print 'Removing masked vertices...'
        remove_vs_indices = [ vi for vi in xrange(len( mesh.vs )) if not mask[ mask.shape[0] - 1 - mesh.vs[vi][1], mesh.vs[vi][0] ] ]
        #print 'remove_vs_indices:'
        #print remove_vs_indices
        mesh.remove_vertex_indices( remove_vs_indices )
        print 'Finished removing masked vertices.'
        '''
        ## Alternative
        def vi_in_mask( vi ): return mask[ mask.shape[0] - 1 - mesh.vs[vi][1], mesh.vs[vi][0] ]
        print 'Removing faces between masked and unmasked vertices...'
        remove_face_indices = [ fi for fi in xrange(len( mesh.faces )) if len(set([ vi_in_mask( vi ) for vi in mesh.faces[fi]])) != 1 ]
        #print 'remove_face_indices:'
        #print remove_face_indices
        mesh.remove_face_indices( remove_face_indices )
        print 'Finished removing faces between masked and unmasked vertices.'
        #'''
    
    return mesh

def grid2vertices_and_faces_and_uvs( grid, four_eight = False ):
    
    assert len( grid.shape ) == 2
    nrows, ncols = grid.shape
    cols, rows = meshgrid( arange( ncols ), arange( nrows ) )
    indices = cols + rows*ncols
    
    vs = asarray( list( zip(
        cols.ravel(),
        ## Flip the 'y' direction that 'rows' runs by subtracting
        ## the row indices from the maximum row index.
        ## NOTE: This behavior is linked to the 'uvs' generation below,
        ##       and also to the assumption about how grid indices
        ##       map to vertex positions, such as in the vertex removal
        ##       according to the 'mask' parameter in 'grid2trimesh()'.
        rows.max() - rows.ravel(),
        grid.ravel()
        ) ) )
    
    uvs = asarray( list( zip(
        cols.ravel()/float(cols.max()),
        rows.ravel()/float(rows.max())
        ) ) )
    
    if not four_eight:
        faces = (
            list( zip(
                indices[:-1,:-1].ravel(),
                indices[1:,:-1].ravel(),
                indices[:-1,1:].ravel()
            ) )
            +
            list( zip(
                indices[:-1,1:].ravel(),
                indices[1:,:-1].ravel(),
                indices[1:,1:].ravel()
            ) )
            )
    else:
        def trieven( oi, oj ):
            return (
                list( zip(
                    indices[oi:-1:2,oj:-1:2].ravel(),
                    indices[oi+1::2,oj:-1:2].ravel(),
                    indices[oi:-1:2,oj+1::2].ravel()
                ) )
                +
                list( zip(
                    indices[oi:-1:2,oj+1::2].ravel(),
                    indices[oi+1::2,oj:-1:2].ravel(),
                    indices[oi+1::2,oj+1::2].ravel()
                ) )
                )
        
        def triodd( oi, oj ):
            return (
                list( zip(
                    indices[oi:-1:2,oj:-1:2].ravel(),
                    indices[oi+1::2,oj+1::2].ravel(),
                    indices[oi::2,oj+1::2].ravel()
                ) )
                +
                list( zip(
                    indices[oi:-1:2,oj:-1:2].ravel(),
                    indices[oi+1::2,oj:-1:2].ravel(),
                    indices[oi+1::2,oj+1::2].ravel()
                ) )
                )
        
        faces = (
            ## even 0, 0
            trieven( 0, 0 )
            +
            ## even 1, 1
            trieven( 1, 1 )
            +
            ## odd 1, 0
            triodd( 1, 0 )
            +
            ## odd 0, 1
            triodd( 0, 1 )
            )
    
    return vs, faces, uvs

def main():
    import sys, os
    
    def usage():
        print >> sys.stderr, "Usage:", sys.argv[0], 'path/to/2D/grid.npy [--clobber] [--4-8] [--z-scale Z] [--normalize|--one-minus-normalize] [--mask /path/to/mask.png] [output.obj]'
        print >> sys.stderr, "--z-scale scales the 2D grid values (default 1.0) after the optional normalize step."
        print >> sys.stderr, "--normalize scales and translates the 2D grid such that the minimum value is 0 and the maximum value is 1."
        print >> sys.stderr, "--one-minus-normalize is the same as normalize, except that it sets values to one minus the normalized value."
        sys.exit(-1)
    
    argv = list( sys.argv )
    
    ## Program name
    del argv[0]
    
    ## Optional arguments
    kClobber = False
    try:
        index = argv.index( '--clobber' )
        del argv[ index ]
        kClobber = True
    except ValueError: pass
    
    four_eight = False
    try:
        index = argv.index( '--4-8' )
        del argv[ index ]
        four_eight = True
    except ValueError: pass
    
    zscale = 1
    try:
        index = argv.index( '--z-scale' )
        try:
            zscale = float( argv[ index+1 ] )
            del argv[ index : index+2 ]
        except:
            usage()
    except ValueError: pass
    
    normalize = False
    try:
        index = argv.index( '--normalize' )
        del argv[ index ]
        normalize = True
    except ValueError: pass
    
    om_normalize = False
    try:
        index = argv.index( '--one-minus-normalize' )
        del argv[ index ]
        om_normalize = True
    except ValueError: pass
    
    mask = None
    try:
        index = argv.index( '--mask' )
        mask_path = argv[ index+1 ]
        del argv[ index : index+2 ]
        import Image
        mask = ( asarray( Image.open( mask_path ).convert('L'), dtype = uint8 ) / 255. ).astype( bool )
        ## This works, but adds dependency on helpers for a one-liner.
        #import helpers
        #mask = helpers.friendly_Image_open_asarray( mask_path ).astype( bool )
        #assert len( mask.shape ) in (2,3)
        #if len( mask.shape ) == 3:
        #    mask = mask.any( axis = 2 )
    except ValueError: pass
    
    ## Input path
    try:
        inpath = argv[0]
        del argv[0]
    except IndexError:
        usage()
    
    ## Output path
    outpath = None
    try:
        outpath = argv[0]
        del argv[0]
    except IndexError:
        outpath = inpath + '.obj'
    
    if normalize and om_normalize:
        print >> sys.stderr, "Can't normalize and one-minus-normalize at the same time."
        usage()
    
    if len( argv ) > 0:
        usage()
    
    if not kClobber and os.path.exists( outpath ):
        print >> sys.stderr, "Output path exists, aborting:", outpath
        sys.exit(-1)
    
    arr = load( inpath )
    assert len( arr.shape ) == 2
    
    ## Debugging
    #import Image
    #Image.fromarray( asarray( ( 255*( arr - arr.min() ) / ( arr.max() - arr.min() ) ).clip(0,255), dtype = uint8 ) ).show()
    ## Test a simple sin/cos surface.
    #arr = zeros((4,5))
    #arr = arr.shape[0]*outer( sin(linspace(0,pi,arr.shape[0])), cos(linspace(0,pi,arr.shape[1])) )
    
    ## Normalize the array.
    if normalize or om_normalize:
        ## Always promote the array's dtype to float if we're normalizing.
        arr = asarray( arr, dtype = float )
        arr = ( arr - arr.min() ) / ( arr.max() - arr.min() )
        if om_normalize: arr = -arr
    
    #arr *= 2
    arr = zscale * arr
    
    save_grid_as_OBJ( arr, outpath, four_eight = four_eight, mask = mask )


def test():
    # import heightmesh
    # heightmesh.test()
    ## or
    # /usr/bin/python2.5 -c 'import heightmesh; heightmesh.test()'
    
    grid = ones( (7,5) )
    mesh_nomask = grid2trimesh( grid )
    
    def print_mesh( mesh ):
        print 'mesh.vs:'
        print asarray( mesh.vs ).round(2)
        print 'mesh.uvs:'
        print asarray( mesh.uvs ).round(2)
        print 'mesh.faces:'
        print mesh.faces
    
    print 'mesh_nomask:'
    print_mesh( mesh_nomask )
    try:
        import trimesh_viewer
        trimesh_viewer.view_mesh( mesh_nomask, 'mesh_nomask' )
    except: pass
    
    mask = ones( grid.shape, dtype = bool )
    mask[3,2] = False
    mask[4,3] = False
    
    print 'mask:'
    print mask
    
    mesh_mask = grid2trimesh( grid, mask = mask )
    print 'mesh_mask:'
    print_mesh( mesh_mask )
    try:
        import trimesh_viewer
        trimesh_viewer.view_mesh( mesh_mask, 'mesh_mask' )
    except: pass

if __name__ == '__main__': main()
