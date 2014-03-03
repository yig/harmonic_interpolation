from numpy import *
import Image
import recovery

def edge_mask2edges( edge_mask ):
    '''
    Given a boolean 2D array 'edge_mask' representing the dual graph
    to a primary graph such that edge_mask[0,0] is taken to be
    primary[ -1/2, -1/2 ],
    returns 'cut_edges' suitable for passing
    to 'gen_symmetric_grid_laplacian()' where adjacent True values
    in 'edge_mask' cut edges in the dual graph.
    The primal graph is assumed to be two smaller in all its dimensions
    (a one-element-wide border).
    
    tested
    >>> edge_mask2edges( array([[True, True, True, True, True], [ False,  False,  True,  False,  False], [False, False,  True, False, False], [False, False,  True, False, False], [False, False,  True, False, False]], dtype=bool ) )
    [((0, 1), (0, 2)), ((1, 1), (1, 2)), ((2, 1), (2, 2)), ((3, 1), (3, 2))]
    >>> edge_mask2edges( array([[False, False, False, False, False], [ True,  True,  True,  True,  True], [False, False,  True, False, False], [False, False,  True, False, False], [False, False,  True, False, False]], dtype=bool ) )
    [((1, 1), (1, 2)), ((2, 1), (2, 2)), ((3, 1), (3, 2)), ((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)), ((0, 3), (1, 3))]
    '''
    
    assert edge_mask.shape[0] > 1
    assert edge_mask.shape[1] > 1
    
    cut_edges = []
    
    assert len( edge_mask.shape ) == 2
    ## Anywhere that the mask and one row down are both true,
    ## add a cut edge between the *column* in the primary graph.
    dual_row_differs = logical_and( edge_mask[:-1,1:-1], edge_mask[1:,1:-1] )
    for dual_i,dual_j in zip( *where( dual_row_differs ) ):
        cut_edges.append( (( dual_i, dual_j ), ( dual_i, dual_j+1 )) )
    
    ## Anywhere that the mask and one column over are both true,
    ## add a cut edge between the *row* in the primary graph.
    dual_col_differs = logical_and( edge_mask[1:-1,:-1], edge_mask[1:-1,1:] )
    for dual_i,dual_j in zip( *where( dual_col_differs ) ):
        cut_edges.append( (( dual_i, dual_j ), ( dual_i+1, dual_j )) )
    
    ## Because of our "1:-1" slice, there should
    ## be no cuts outside the primal graph.
    if len( cut_edges ) > 0:
        assert asarray([ ( e[0][0], e[1][0] ) for e in cut_edges ]).min() >= 0
        assert asarray([ ( e[0][0], e[1][0] ) for e in cut_edges ]).max() < edge_mask.shape[0]-1
        assert asarray([ ( e[0][1], e[1][1] ) for e in cut_edges ]).min() >= 0
        assert asarray([ ( e[0][1], e[1][1] ) for e in cut_edges ]).max() < edge_mask.shape[1]-1
    
    return cut_edges

def main():
    import surface_with_edges_helpers as helpers
    
    ### Globals ###
    #centroids2normals_path = 'HITs3-normal_map_viz.centroids2normals'
    centroids2normals_path = 'HITs3-siga-normal_map_viz.centroids2normals'
    
    image_path = 'buddhist_column_at_sarnath-small.png'
    #mask_path = 'buddhist_column_at_sarnath-small-mask.png'
    mask_path = None
    #cut_edges_mask_path = 'buddhist_column_at_sarnath-small-patches-color2-flat-edges-thin-culled-red_disconnected-green_sharp.png'
    #cut_edges_mask_path = 'buddhist_column_at_sarnath-small-patches-color2-flat-edges-thin-culled-green_sharp.png'
    cut_edges_mask_path = 'buddhist_column_at_sarnath-small-patches-color2-flat-edges-thin-culled-red_disconnected.png'
    
    output_image_path = 'surface_with_edges'
    output_image_path_surface = helpers.unique( output_image_path + '-surface' + '.png' )
    output_npy_path_surface = helpers.unique( output_image_path + '-surface' + '.npy' )
    output_obj_path_surface = helpers.unique( output_image_path + '-surface' + '.obj' )
    
    ## ============= ## ============= ## ============= ## ============= ##
    
    ### Load files ###
    
    rows, cols = helpers.friendly_Image_open_asarray( image_path ).shape[:2]
    
    dx = []
    dy = []
    cut_edges = []
    value_constraints = [(0,0,0)]
    
    mask = None
    if mask_path is not None:
        mask = helpers.friendly_Image_open_asarray( mask_path ).astype( bool )
        assert len( mask.shape ) in (2,3)
        if len( mask.shape ) == 3:
            mask = mask.any( axis = 2 )
        
        cut_edges.extend( recovery.cut_edges_from_mask( mask ) )
        
        ## We need to make sure there is a value constraint for each connected component.
        if len( cut_edges ) > 0:
            from segm2unique import segm2unique
            from segmentation_utils import segmentation2regions
            greymask = asarray( mask, dtype = uint8 )
            connected_components = segm2unique( greymask )
            ## regions are in numpy.where() format.
            regions = segmentation2regions( connected_components )
            value_constraints = [ ( r[0][0], r[1][0], 0. ) for r in regions ]
    
    ignore_mask = zeros( ( rows, cols ), dtype = bool )
    if cut_edges_mask_path is not None:
        cut_edges_mask = friendly_Image_open_asarray( cut_edges_mask_path )
        disconnected_mask = ( cut_edges_mask == ( 255, 0, 0 ) ).all( axis = 2 )
        discontinuous_mask = ( cut_edges_mask == ( 0, 255, 0 ) ).all( axis = 2 )
        
        ## Don't add normals within 10 pixels of these constraints.
        kProtectRadius = 10
        from highlighting import grow
        ignore_mask = logical_or( ignore_mask, grow( disconnected_mask, kProtectRadius ) )
        ignore_mask = logical_or( ignore_mask, grow( discontinuous_mask, kProtectRadius ) )
        
        ## TODO: disconnected_mask could induce more disconnected components,
        ##       but I have no way of checking for or dealing with it.
        cut_edges.extend( edge_mask2edges( disconnected_mask ) )
        
        discontinuous_cut_edges = edge_mask2edges( discontinuous_mask )
        dx_disc, dy_disc = recovery.zero_dx_dy_constraints_from_cut_edges( discontinuous_cut_edges )
        
        cut_edges.extend( discontinuous_cut_edges )
        
        '''
        ## Overwrite values in dx, dy if there are duplicates.
        dx2val = dict([ ( ( row, col ), val ) for row, col, val in dx ])
        dy2val = dict([ ( ( row, col ), val ) for row, col, val in dy ])
        dx2val.update( [ ( ( row, col ), val ) for row, col, val in dx_disc ] )
        dy2val.update( [ ( ( row, col ), val ) for row, col, val in dy_disc ] )
        dx = [ ( row, col, val ) for ( row, col ), val in dx2val.iteritems() ]
        dy = [ ( row, col, val ) for ( row, col ), val in dy2val.iteritems() ]
        '''
        dx.extend( dx_disc )
        dy.extend( dy_disc )
        #'''
    
    
    from unique2centroids import lines2centroids2normals
    row_col2normal = lines2centroids2normals( open( centroids2normals_path ) )
    ## Transpose normal from right, down to row, col.
    row_col2normal = dict([ ( row_col, ( normal[1], normal[0], normal[2] ) ) for row_col, normal in row_col2normal.iteritems() ])
    
    #normal_constraints = [ ( row, col, normal ) for ( row, col ), normal in row_col2normal.iteritems() ]
    normal_constraints = [
        ( row, col, normal )
        for ( row, col ), normal in row_col2normal.iteritems()
        if not ignore_mask[ row, col ]
        ]
    
    ndx, ndy = recovery.convert_grid_normal_constraints_to_dx_and_dy_constraints( rows, cols, normal_constraints )
    
    ## Don't overwrite values in dx, dy if there are duplicates.
    #dx2val = dict([ ( ( row, col ), val ) for row, col, val in dx ])
    #dy2val = dict([ ( ( row, col ), val ) for row, col, val in dy ])
    #dx2val.update( [ ( ( row, col ), val ) for row, col, val in dx_disc ] )
    #dy2val.update( [ ( ( row, col ), val ) for row, col, val in dy_disc ] )
    #dx = [ ( row, col, val ) for ( row, col ), val in dx2val.iteritems() ]
    #dy = [ ( row, col, val ) for ( row, col ), val in dy2val.iteritems() ]
    ## UPDATE: There shouldn't be duplicates because of 'ignore_mask'
    #dx.extend( ndx )
    #dy.extend( ndy )
    ## What if we use the edges only to clear normals near edges?
    cut_edges = []
    dx = ndx
    dy = ndy
    
    
    nmap = recovery.solve_grid_linear( rows, cols, dx_constraints = dx, dy_constraints = dy, value_constraints = value_constraints, bilaplacian = True, cut_edges = cut_edges )
    ## scale by 2.
    nmap *= 2.
    
    ## Make sure we get the same thing without looking at 'cut_edges_mask_path'.
    ## We do.
    #from normal_map_viz import generate_surface_from_mask_and_normals
    #nmap2 = generate_surface_from_mask_and_normals( mask, row_col2normal )
    #debugger()
    
    ## I want to be very explicit and call numpy.save(),
    ## in case I imported something else with the name 'save'.
    import numpy
    numpy.save( output_npy_path_surface, nmap )
    print '[Saved surface depth map as a numpy array to "%s".]' % (output_npy_path_surface,)
    
    nmap_arr = recovery.normalize_to_char_img( nmap )
    nmap_img = Image.fromarray( nmap_arr )
    nmap_img.save( output_image_path_surface )
    print '[Saved surface depth map as an image to "%s".]' % (output_image_path_surface,)
    nmap_img.show()
    
    import heightmesh
    heightmesh.save_grid_as_OBJ( nmap, output_obj_path_surface, mask = mask )
    print '../GLUTViewer "%s" "%s"' % ( output_obj_path_surface, image_path )
    
    #debugger()
    pass

if __name__ == '__main__': main()
