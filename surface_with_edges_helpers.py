def friendly_Image_open_asarray( filename ):
    from numpy import asarray, uint8
    from PIL import Image
    img = Image.open( filename )
    ## Merge down to a greyscale (Luminance) image.
    ## UPDATE: Actually, don't do this.  We want to work on a color image!
    #img = img.convert( 'L' )
    ## Convert back and forth to numpy array with numpy.asarray( img ) and Image.fromarray( arr )
    ## Source: http://stackoverflow.com/questions/384759/pil-and-numpy
    ## One-liner: arr = asarray( img )
    ## One-liner: Image.fromarray( arr ).save( 'foo.png' )
    ## One-liner: Image.fromarray( arr ).show()
    ## One-liner (back and forth): Image.fromarray( asarray( img, dtype = uint8 ) ).save( 'foo.png' )
    ## One-liner (back and forth): Image.fromarray( asarray( img, dtype = uint8 ) ).show()
    #arr = asarray( img ) / 255.
    arr = asarray( img, dtype = uint8 )
    ## Ignore the alpha channel if there is one.
    assert len( arr.shape ) == 2 or ( len( arr.shape ) == 3 and arr.shape[2] in (3,4) )
    if len( arr.shape ) == 3: arr = arr[:,:,:3]
    
    return arr

def unique( in_fname ):
    import os
    
    orig_fname = in_fname.rstrip('/')
    
    count = 1
    fname = orig_fname
    
    while os.path.exists( fname ):
        fname = os.path.splitext( orig_fname )[0] + ' ' + str(count) + os.path.splitext( orig_fname )[1]
        count += 1
    
    return fname
