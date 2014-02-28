Dependencies:
    Python >= 2.6 and < 3
    PIL (Python Image Library)
    scipy
    cvxopt

Look into modifying surface_with_edges.py main().
It calls recovery.py solve_grid_linear().

To smooth bumps created by normals constraints, try calling
recovery.py smooth_bumps() with the locations of the bumps and
some integer value for bump_radius.
