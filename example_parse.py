def parse_xynormal_file_path( file_path ):
    '''
    If your file has the format:
    
    x y normalx normaly normalz
    
    where x and y are integers and normal{x,y,z} are floating-point values,
    then this will parse it,
    returning a list where each element is a tuple ( x, y, normalx, normaly, normalz ).
    '''
    
    xynormal = []
    
    f = open( file_path )
    for line in f:
        sline = line.strip().split()
        x = int( sline[0] )
        y = int( sline[1] )
        nx = float( sline[2] )
        ny = float( sline[3] )
        nz = float( sline[4] )
        
        xynormal.append( ( x, y, nx, ny, nz ) )
    
    return xynormal
