from numpy import *
from PIL import Image
from itertools import izip as zip
from recovery import solve_grid_linear, solve_grid_linear_simple

def fill_alpha_harmonic( arr, mask ):
    
    #from pydb import debugger
    #debugger()
    
    ### 1 Generate laplacian matrix for arr.
    ### 2 Set constraints so that locations where 'mask' are True stay the same.
    ### 3 Solve for the remainder.
    
    ## solve_grid_linear_simple() requires vector-values constraints:
    if len( arr.shape ) == 2: arr = arr[...,newaxis]
    
    return solve_grid_linear_simple(
        arr.shape[0],
        arr.shape[1],
        value_constraints = [
            ( i, j, arr[ i,j ] )
            for ( i,j ) in zip( *where( mask ) )
            ]
        )

def load_img2arr( path ):
    '''
    Given a 'path' to an image,
    returns a numpy.array (rows-by-columns-by-RGBA)
    containing the image with floating point values between 0 and 1.
    '''
    
    img = Image.open( path ).convert( 'RGBA' )
    arr = asarray( img, dtype = uint8 ) / 255.
    return arr.astype( float )

def arr2img( arr ):
    '''
    Given a numpy.array 'arr' representing a floating point image (one whose values are between 0 and 1),
    returns the conversion of 'arr' into a PIL Image.
    '''
    
    assert arr.dtype in ( float32, float64, float )
    return Image.fromarray( asarray( ( arr * 255 ).round(0).clip( 0, 255 ), dtype = uint8 ) )

def main():
    import os, sys
    
    def usage():
        print >> sys.stderr, 'Usage:', sys.argv[0], 'path/to/input_image path/to/output_image'
        
        print >> sys.stderr, 'Replaces pixels whose opacity is not 100% with a harmonic function. 100% opaque pixels remain unchanged.'
        print >> sys.stderr, 'Example:', sys.argv[0], '"fill_alpha_harmonic-test/test.png" "fill_alpha_harmonic-test/out.png"'
        print >> sys.stderr, 'NOTE: To verify, use ql "fill_alpha_harmonic-test/"*.png'
        
        sys.exit(-1)
    
    try:
        inpath, outpath = sys.argv[1:]
    except:
        usage()
    
    if not os.path.isfile( inpath ):
        print >> sys.stderr, 'ERROR: Input path does not exist.'
        usage()
    if os.path.exists( outpath ):
        print >> sys.stderr, 'ERROR: Output path exists; not clobbering.'
        usage()
    
    print 'Loading input image:', inpath
    arr = load_img2arr( inpath )
    
    print 'Filling in non-opaque values...'
    result = fill_alpha_harmonic( arr[:,:,:3], ( 255*arr[:,:,3] ).round(0).astype( int ) == 255 )
    result_img = arr2img( result )
    
    result_img.save( outpath )
    print 'Saved output to:', outpath


## UPDATE: If I don't encapsulate the test inside a function, I can run this
##         script with "python -i" to investigate the variables.
#def test():
kTest = False
if kTest:
    img = ones( ( 9, 9, 4 ), dtype = uint8 )
    img *= 255
    
    kTest = 'upperleft'
    if 'upperleft' == kTest:
        img[:3,:3,3] = 0
    elif 'center' == kTest:
        img[3:6,3:6,3] = 0
    
    arr = img / 255.
    result = fill_alpha_harmonic( arr[:,:,:3], ( 255*arr[:,:,3] ).round(0).astype( int ) == 255 )
    
    print allclose( result, arr[:,:,:3] )
else:
    if __name__ == '__main__': main()
