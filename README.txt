Dependencies:
    Python >= 2.6 and < 3
    PIL (Python Image Library)
    scipy
    cvxopt

For diffusing scalar values with hard and soft constraints:
    Call recovery.solve_grid_linear_simple2().
    You can look at recovery.test_solve_grid_linear_simple2() for an example.
    If you run
        python recovery.py
    it will call solve_grid_linear_simple2() and generate test output.

For recovering a height field:
    Look into modifying surface_with_edges.py main().
    It calls recovery.py solve_grid_linear().
    
    To smooth bumps created by normals constraints, try calling
    recovery.py smooth_bumps() with the locations of the bumps and
    some integer value for bump_radius.
