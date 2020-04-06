from numpy import *


def array_outlining_span( img, span, color ):
    '''
    Given an image 'img', a span 'region' of four integer values
    specifying a rectangular region 'img[ region[0]:region[1], region[2]:region[3] ]',
    and a tuple of three rgb values 'color',
    returns a copy of 'img' with the given region outlined in a border of color 'color'.
    '''
    from PIL import Image, ImageDraw
    ## Call img.copy() because I'm not positive that Image.fromarray() makes a copy.
    assert img.dtype == uint8
    #img = asarray( img, dtype = uint8 )
    assert len( img.shape ) == 3
    assert img.shape[2] == 3
    result_img = Image.fromarray( img.copy(), mode = 'RGB' )
    draw = ImageDraw.Draw( result_img )
    y0, y1, x0, x1 = span
    draw.line( [ (x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0) ], width = 3, fill = color )
    del draw
    #result_img.show()
    #debugger()
    return asarray( result_img, dtype = uint8 )

def shrink( mask, pixel_iterations ):
    return logical_not( grow( logical_not( mask ), pixel_iterations ) )
def grow( mask, pixel_iterations ):
    '''
    Given a 2D boolean numpy.array 'mask' and number of iterations 'pixel_iterations',
    Returns a 2D array created by expanding the True area of 'mask' by 'pixel_iterations' pixels.
    '''
    
    assert int( pixel_iterations ) == pixel_iterations
    assert pixel_iterations >= 0
    
    ## Copy the input array.
    mask = array( mask )
    ## Smear in x and y over and over; each loop expands the region by one pixel.
    for i in range( pixel_iterations ):
        '''
        ## This doesn't work due to aliasing.
        mask[ :-1, : ] += mask[ 1:, : ]
        mask[ 1:, : ] += mask[ :-1, : ]
        mask[ :, :-1 ] += mask[ :, 1: ]
        mask[ :, 1: ] += mask[ :, :-1 ]
        '''
        
        #'''
        ## This works, and makes diamond corners.
        mask_copy = array( mask )
        mask[ :-1, : ] += mask_copy[ 1:, : ]
        mask[ 1:, : ] += mask_copy[ :-1, : ]
        mask[ :, :-1 ] += mask_copy[ :, 1: ]
        mask[ :, 1: ] += mask_copy[ :, :-1 ]
        #'''
        
        '''
        ## This works, and makes sharp corners.
        mask[ :-1, : ] += array( mask[ 1:, : ] )
        mask[ 1:, : ] += array( mask[ :-1, : ] )
        mask[ :, :-1 ] += array( mask[ :, 1: ] )
        mask[ :, 1: ] += array( mask[ :, :-1 ] )
        '''
        
        '''
        ## This works, and makes sharp corners.
        mask[ :-1, : ] = mask[ :-d1, : ] + mask[ 1:, : ]
        mask[ 1:, : ] = mask[ 1:, : ] + mask[ :-1, : ]
        mask[ :, :-1 ] = mask[ :, :-1 ] + mask[ :, 1: ]
        mask[ :, 1: ] = mask[ :, 1: ] + mask[ :, :-1 ]
        '''
    
    return mask

def array_outlining_region( img, region, color ):
    '''
    Given an image 'img', a region 'region' of coordinates
    specifying a region 'img[ region ]',
    and a tuple of three rgb values 'color',
    returns a copy of 'img' with the given region outlined in a border of color 'color'.
    '''
    
    assert img.dtype == uint8
    ## NOTE: We are assuming an rgb image, though we don't actually need one.
    ##       Really all we need is to be able to do img[i,j] = color.
    assert len( img.shape ) == 3
    assert img.shape[2] == 3
    assert len( color ) == 3
    
    mask = zeros( img.shape[:2], dtype = bool )
    
    mask[ region ] = True
    mask = grow( mask, 6 )
    
    ## Remove the interior
    mask[ region ] = False
    
    ## Make a copy, fill in the mask.
    result = array( img )
    result[ mask ] = color
    
    #from PIL import Image
    #Image.fromarray( asarray( result, dtype = uint8 ) ).show()
    #debugger()
    return result

def array_tinting_region( img, region, color, alpha ):
    '''
    Given an image 'img', a region 'region' of coordinates
    specifying a region 'img[ region ]',
    a tuple of three rgb values 'color',
    and a blend value 'alpha' in the range [0,1],
    returns a copy of 'img' with the given region tinted
    by 'color' where a value of 'alpha' = 0 means no change
    and a value of 'alpha' = 1 means the region is replaced with 'color'.
    '''
    
    assert img.dtype == uint8
    ## NOTE: We are assuming an rgb image, though we don't actually need one.
    ##       Really all we need is to be able to do img[i,j] = color.
    assert len( img.shape ) == 3
    assert img.shape[2] == 3
    assert len( color ) == 3
    
    alpha = float( alpha )
    alpha = max( 0., min( 1., alpha ) )
    
    ## Make a copy.
    result = array( img )
    ## Blend the area specified by 'region' with 'color' according to 'alpha', linearly.
    result[ region ] = ( img[ region ] + alpha * ( color - img[ region ] ) ).round().astype( uint8 ).clip( 0, 255 )
    
    #from PIL import Image
    #Image.fromarray( asarray( result, dtype = uint8 ) ).show()
    #debugger()
    return result

def array_tinting_region_with_falloff_from_center( img, region, color, alpha ):
    '''
    Given an image 'img', a region 'region' of coordinates
    specifying a region 'img[ region ]',
    a tuple of three rgb values 'color',
    and a blend value 'alpha' in the range [0,1],
    returns a copy of 'img' with the given region
    tinted by 'color' such that the tint is strongest in the
    center of the region and falls off at the boundary.
    A value of 'alpha' = 1 means the region is replaced with 'color' at its center
    and a value of 'alpha' = 0 means no change anywhere.
    '''
    
    assert img.dtype == uint8
    ## NOTE: We are assuming an rgb image, though we don't actually need one.
    ##       Really all we need is to be able to do img[i,j] = color.
    assert len( img.shape ) == 3
    assert img.shape[2] == 3
    assert len( color ) == 3
    
    alpha = float( alpha )
    alpha = max( 0., min( 1., alpha ) )
    
    
    kCropAndBuffer = True
    if not kCropAndBuffer:
        r = zeros( img.shape[:2], dtype = bool )
        r[ region ] = True
    else:
        ## Only solve a crop around the region.
        ## Add a buffer pixel so I can have a proper boundary conditions around
        ## the region even if it reaches all the way to the edge of the image.
        region_mask = zeros( img.shape[:2], dtype = bool )
        region_mask[ region ] = True
        
        kBufferPixels = 1
        crop = region[0].min(), region[0].max()+1, region[1].min(), region[1].max()+1
        r = zeros( ( crop[1]-crop[0] + kBufferPixels*2, crop[3]-crop[2] + kBufferPixels*2 ), dtype = bool )
        r[ kBufferPixels:-kBufferPixels, kBufferPixels:-kBufferPixels ][ region_mask[ crop[0] : crop[1], crop[2] : crop[3] ] ] = True
    
    
    ## Solve a bilaplacian system to make a bump inside the region.
    router = grow( r, 1 )
    router[ r ] = False
    rinner = logical_not( grow( logical_not( r ), 1 ) )
    rim = r - rinner
    ## NOTE: I used to constrain 'rim' to '0' and 'router' to '-1', but that leads to a divide-by-zero
    ##       if the region is one or two pixels wide.
    value_constraints = [ (i,j,1.) for i,j in zip(*where(rim)) ] + [ (i,j,0.) for i,j in zip(*where(router)) ]
    print('len( value_constraints ):', len( value_constraints ))
    from recovery import solve_grid_linear
    bump = solve_grid_linear( r.shape[0], r.shape[1], bilaplacian = True, value_constraints = value_constraints, iterative = False )
    ## Mask the bump outside the region.
    bump[ logical_not( r ) ] = 0.
    assert bump.min() >= -1e-5
    ## Normalize the bump so the maximum value is 1.
    bump *= 1./bump.max()
    ## Guarantee it despite floating point rounding.
    bump = bump.clip(0.,1.)
    
    ## Make a copy of the input image.
    result = array( img )
    ## Blend the area specified by 'region' with 'color' according to 'alpha' and 'bump', linearly.
    bump *= alpha
    
    if not kCropAndBuffer:
        result[ region ] = ( img[ region ] + bump[ region ][ :, newaxis ] * ( color - img[ region ] ) ).round().astype( uint8 ).clip( 0, 255 )
    else:
        result_slice = result[ crop[0] : crop[1], crop[2] : crop[3] ]
        result_slice[:] = ( result_slice + bump[ kBufferPixels : -kBufferPixels, kBufferPixels : -kBufferPixels, newaxis ] * ( color - result_slice ) ).round().astype( uint8 ).clip( 0, 255 )
    
    #from PIL import Image
    #Image.fromarray( asarray( result, dtype = uint8 ) ).show()
    #debugger()
    return result

def array_outlining_regions_border( img, regions, colors, pixels_from_border = None, outline_pixel_thickness = None ):
    '''
    Given an image 'img', a sequence of two abutting regions 'regions' of
    coordinates specifying a region 'img[ regions[r] ]',
    and a sequence of two tuples of three rgb values 'colors',
    returns a copy of 'img' with the border between the two regions
    outlined with a border on each side colored with the corresponding color.
    
    WARNING: This algorithm exhibited some strange behavior when one
             region was entirely surrounded by the other; I don't know why and
             I haven't investigated.  array_tinting_regions_borders() behaved
             fine.
    '''
    
    assert img.dtype == uint8
    ## NOTE: We are assuming an rgb image, though we don't actually need one.
    ##       Really all we need is to be able to do img[i,j] = color.
    assert len( img.shape ) == 3
    assert img.shape[2] == 3
    assert len( set( [ id(r) for r in regions ] ) ) == len( regions )
    assert False not in [ len( color ) == 3 for color in colors ]
    
    if pixels_from_border is None: pixels_from_border = 6
    if outline_pixel_thickness is None: outline_pixel_thickness = 3
    assert int( pixels_from_border ) == pixels_from_border
    assert int( outline_pixel_thickness ) == outline_pixel_thickness
    assert pixels_from_border > 0
    assert outline_pixel_thickness > pixels_from_border
    
    ## Disclaimer: Although the following is written as if it works for an
    ##             arbitrary number of regions, it has only been reasoned
    ##             about when there are two regions.
    ## 1 Grow each region, then intersect the grown region with the union of all other regions.
    ## 2 Take the union of all of the intersections from step 1.
    ## 3 Shrink it (grow the complement).
    ## 4 Take the product of step 2 and remove the product of step 3.
    ## 5 For each region, paint the intersection of the region with the output of step 4 the region's color.
    
    
    ## 1
    step2 = zeros( img.shape[:2], dtype = bool )
    for region in regions:
        region_mask = zeros( img.shape[:2], dtype = bool )
        region_mask[ region ] = True
        region_mask = grow( region_mask, pixels_from_border )
        
        other_mask = zeros( img.shape[:2], dtype = bool )
        for other in regions:
            if other is region: continue
            other_mask[ other ] = True
        step1 = logical_and( region_mask, other_mask )
        
        ## 2
        step2[ step1 ] = True
    
    ## 3
    step3 = logical_not( step2 )
    step3 = grow( step3, outline_pixel_thickness )
    
    ## 4
    step4 = logical_and( step2, step3 )
    
    ## 5
    ## Make a copy.
    result = array( img )
    for region, color in zip( regions, colors ):
        region_mask = zeros( img.shape[:2], dtype = bool )
        region_mask[ region ] = True
        result[ logical_and( region_mask, step4 ) ] = color
    
    #from PIL import Image
    #Image.fromarray( asarray( result, dtype = uint8 ) ).show()
    #debugger()
    return result

def array_tinting_regions_borders( img, regions, colors, alpha, pixels_from_border = None, highlighted_region_out = None ):
    '''
    Given an image 'img', a sequence of two abutting regions 'regions' of
    coordinates specifying a region 'img[ regions[r] ]',
    a sequence of two tuples of three rgb values 'colors',
    and a blend value 'alpha' in the range [0,1],
    returns a copy of 'img' with the area around the border between the
    regions tinted by the color corresponding to the region whose territory
    it is; a value of 'alpha' = 0 means no change and a value of 'alpha' = 1
    means the region is replaced with 'color'.
    '''
    
    assert img.dtype == uint8
    ## NOTE: We are assuming an rgb image, though we don't actually need one.
    ##       Really all we need is to be able to do img[i,j] = color.
    assert len( img.shape ) == 3
    assert img.shape[2] == 3
    assert len( set( [ id(r) for r in regions ] ) ) == len( regions )
    assert False not in [ len( color ) == 3 for color in colors ]
    
    alpha = float( alpha )
    alpha = max( 0., min( 1., alpha ) )
    
    if pixels_from_border is None: pixels_from_border = 6
    assert int( pixels_from_border ) == pixels_from_border
    assert pixels_from_border > 0
    
    ## Disclaimer: Although the following is written as if it works for an
    ##             arbitrary number of regions, it has only been reasoned
    ##             about when there are two regions.
    ## 1 Grow each region, then intersect the grown region with the union of all other regions.
    ## 2 Take the union of all of the intersections from step 1.
    ## 3 For each region, paint the intersection of the region with the output of step 4 the region's color.
    
    
    ## 1
    step2 = zeros( img.shape[:2], dtype = bool )
    for region in regions:
        region_mask = zeros( img.shape[:2], dtype = bool )
        region_mask[ region ] = True
        region_mask = grow( region_mask, pixels_from_border )
        
        other_mask = zeros( img.shape[:2], dtype = bool )
        for other in regions:
            if other is region: continue
            other_mask[ other ] = True
        step1 = logical_and( region_mask, other_mask )
        
        ## 2
        step2[ step1 ] = True
    
    ## 3
    ## Make a copy.
    result = array( img )
    for region, color in zip( regions, colors ):
        region_mask = zeros( img.shape[:2], dtype = bool )
        region_mask[ region ] = True
        
        mask = logical_and( region_mask, step2 )
        ## Blend the area specified by 'mask' with 'color' according to 'alpha', linearly.
        result[ mask ] = ( result[ mask ] + alpha * ( color - result[ mask ] ) ).round().clip( 0, 255 ).astype( uint8 )
    
    if highlighted_region_out is not None:
        del highlighted_region_out[:]
        highlighted_region_out.append( where( step2 ) )
    
    #from PIL import Image
    #Image.fromarray( asarray( result, dtype = uint8 ) ).show()
    #debugger()
    return result

def array_tinting_region_border_with_falloff( img, regions, colors, alpha ):
    '''
    Given an image 'img', a sequence of two abutting regions 'regions' of
    coordinates specifying a region 'img[ regions[r] ]',
    a sequence of two tuples of three rgb values 'colors',
    and minimum and maximum blend values 'min_alpha' and 'max_alpha'
    in the range [0,1],
    returns a copy of 'img' with the regions tinted such that they each are
    tinted by their corresponding colors most strongly at their shared border
    and gradually falling off to the untinted value at the rest of their boundaries;
    the value 'alpha' = 1 means the tint is opaque at the shared border,
    while a value of 'alpha' = 0 means no tinting occurs.
    '''
    
    #debugger()
    
    assert img.dtype == uint8
    ## NOTE: We are assuming an rgb image, though we don't actually need one.
    ##       Really all we need is to be able to do img[i,j] = color.
    assert len( img.shape ) == 3
    assert img.shape[2] == 3
    assert len( set( [ id(r) for r in regions ] ) ) == len( regions )
    assert False not in [ len( color ) == 3 for color in colors ]
    
    alpha = float( alpha )
    alpha = max( 0., min( 1., alpha ) )
    
    
    ## 1 Grow each region by one pixel.
    ## 2 For each region, solve a laplace equation where values
    ##   are zero outside the region and one at points that are both
    ##   inside the region and inside one or more other grown regions.
    ## 3 Blend according to the solution to the laplace equation.
    
    ## Make a copy.
    result = array( img )
    
    ## 1
    masks = []
    for region in regions:
        mask = zeros( img.shape[:2], dtype = bool )
        mask[ region ] = True
        masks.append( mask )
    
    grown_masks = [ grow( mask, 1 ) for mask in masks ]
    
    ## 2
    kBufferPixels = 2
    for region, mask, grown_mask, color in zip( regions, masks, grown_masks, colors ):
        ## This is the smaller region inside the image that we are looking at.
        crop = region[0].min(), region[0].max()+1, region[1].min(), region[1].max()+1
        ## This is the mask pasted onto a canvas of zeros with a two-pixel border.
        crop_mask = zeros( ( crop[1]-crop[0] + kBufferPixels*2, crop[3]-crop[2] + kBufferPixels*2 ), dtype = bool )
        crop_mask[ kBufferPixels:-kBufferPixels, kBufferPixels:-kBufferPixels ][ mask[ crop[0] : crop[1], crop[2] : crop[3] ] ] = True
        assert len( where( crop_mask ) ) == len( where( mask ) )
        
        ## 'grown_mask - mask' gives us just the one-ring around make, but we need the two ring.
        ## For now, just constrain everything outside the region.
        #zero_constraints = set([ (i,j,0) for i,j in zip( *where( grown_mask - mask ) ) ])
        #zero_constraints = set([ (i,j) for i,j in zip( *where( logical_not( mask ) ) ) ])
        zero_constraints = set([ (i,j) for i,j in zip( *where( logical_not( crop_mask ) ) ) ])
        
        one_constraints = set()
        for other, other_mask, other_grown_mask in zip( regions, masks, grown_masks ):
            if other is region: continue
            ## This may create duplicate entries in one_constraints, so we'll make it a set.
            ## Q: Should I use 'mask' or 'grown'mask to logical_and() with 'other_grown_mask'?
            ## A1: Use 'mask', so I get only pixels inside 'region'.
            ## A2: Use 'grown_mask', so I get a border of pixels with
            ##     thickness 2, which is important for the boundary
            ##     constraints in the laplacian system.
            ## A2 UPDATE: There will now be a conflict between some of the
            ##            zero_constraints and the one_constraints that come
            ##            from the grown part of grown_mask.
            one_constraints.update( [ (i-crop[0]+kBufferPixels,j-crop[2]+kBufferPixels) for i,j in zip( *where( logical_and( grown_mask, other_grown_mask ) ) ) ] )
        
        ## Because we're using grown_mask for one_constraints, there
        ## will be conflict between one_constraints and zero_constraints.
        #assert len( one_constraints.intersection( zero_constraints ) ) == 0
        assert len( one_constraints.intersection( zero_constraints ) ) > 0
        ## Remove the conflicting elements from 'zero_constraints'.
        zero_constraints = zero_constraints.difference( one_constraints )
        
        ## Solve the laplace equation.
        ## TODO: Shrink down to a bounding box around 'region' to avoid
        ##       solving a *much* larger system than necessary.
        ##       Something like: bounds = min( region[0] ), max( region[0] ), min( region[1] ), max( region[1] )
        ##       and then taking a slice of 'bounds' after expanding it on
        ##       either side by 2, putting that in a smaller array,
        ##       solving there, then using that to modify a subregion of 'result'.
        from recovery import solve_grid_linear
        ## Q: Laplacian or Bilaplacian?
        ## A: Laplacian.  They both produce similar results but bilaplacian is slower (*much* slower with an iterative method, which we're using).
        blend = solve_grid_linear( crop_mask.shape[0], crop_mask.shape[1], value_constraints = [ (i,j,0.) for i,j in zero_constraints ] + [ (i,j,1.) for i,j in one_constraints ], bilaplacian = False )
        ## Clip blend to the area within 'region'.
        blend[ logical_not( crop_mask ) ] = 0.
        blend *= alpha
        
        ## 3
        #debugger()
        result_slice = result[ crop[0] : crop[1], crop[2] : crop[3] ]
        result_slice[:] = ( result_slice + blend[ kBufferPixels : -kBufferPixels, kBufferPixels : -kBufferPixels, newaxis ] * ( color - result_slice ) ).round().astype( uint8 ).clip( 0, 255 )
    
    return result

def array_desaturating( img, desaturation ):
    '''
    Given an image 'img',
    and a desaturation value 'desaturation' in the range [0,1],
    where 'desaturation == 0' does not change the image
    and 'desaturation == 1' returns a greyscale image,
    returns a copy of 'img' with its saturation affected by 'desaturation'
    where 'desaturation == 0' does not change the image
    and 'desaturation == 1' returns a greyscale image,
    '''
    
    img = asarray( img )
    
    assert img.dtype == uint8
    ## NOTE: We are assuming an rgb image, though we don't actually need one.
    ##       Really all we need is to be able to do img[i,j] = color.
    assert len( img.shape ) == 3
    assert img.shape[2] == 3
    
    desaturation = float( desaturation )
    desaturation = max( 0., min( 1., desaturation ) )
    
    ## Greyscale image.
    ## Use weights for RGB according to:
    ## http://stackoverflow.com/questions/687261/converting-rgb-to-grayscale-intensity
    ## http://www.mathworks.com/access/helpdesk/help/toolbox/images/rgb2gray.html
    grey = average( img, axis = 2, weights = ( 0.2989, 0.5870, 0.1140 ) )[:,:,newaxis]
    
    ## Blend the two images according to desaturation.
    result = ( img + desaturation * ( grey - img ) ).round().astype( uint8 ).clip( 0, 255 )
    
    #from PIL import Image
    #Image.fromarray( asarray( result, dtype = uint8 ) ).show()
    #debugger()
    return result
