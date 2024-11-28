This directory has three main components:
- A way to construct environments ([nonasymptotic/envs.py](/nonasymptotic/envs.py)).
- Multiple implementations of the radius and PRM algorithms [^1] ([nonasymptotic/prm.py](/nonasymptotic/prm.py)).
- Numerical computations of the sample-complexity bound (e.g. Algorithm 2 in the paper)
  ([nonasymptotic/prm.py](/nonasymptotic/bound.py)).

[^1] Karaman S, Frazzoli E. Sampling-based algorithms for optimal motion planning. 
The International Journal of Robotics Research. 2011;30(7):846-894.