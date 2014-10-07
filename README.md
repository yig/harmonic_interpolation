## For diffusing scalar values with hard and soft constraints and possibly disconnected (cut) edges

Call `recovery.solve_grid_linear_simple3()` or `recovery.solve_grid_linear_simple3_solver()`.
You can look at `recovery.test_solve_grid_linear_simpleN()` for an example.
If you run
    `python recovery.py`
it will call solve_grid_linear_simple3() and generate test output.


## For recovering a height field

Look into modifying `surface_with_edges.py` `main()`.
It calls `recovery.solve_grid_linear()`.
(It relies on lots of external helper function to load its input, which isn't necessary.)

To smooth bumps created by normals constraints, try calling
`recovery.smooth_bumps()` or `recovery.smooth_bumps2()` with the locations of the bumps and
some integer value for bump_radius.


## For filling in holes in an image harmonically

To replace pixels in an image whose opacity is not 100% with a harmonic function:

        python fill_alpha_harmonic.py path/to/image_with_non_opaque_pixels.png path/to/output.png

100% opaque pixels remain unchanged.


## Dependencies

* Python >= 2.6 and < 3
* PIL (Python Image Library) (only needed for the examples and tests)
* scipy
* cvxopt
