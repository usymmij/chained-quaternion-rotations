# Rotations
this python module is used for generating 3D vectors, and rotating them using quaternions with hardware acceleration using torch cuda
- originally written for generating a database of chained rotations for statistical ML regression
- chained rotations are vulnerable to gimbal lock so quaternions are used 
- used torch for GPU acceleration, although in the end it turned out it runs faster on CPU for my purposes: GPU is faster when more rotations are chained 
	as it accelerates rotation calculations, but not random generation
