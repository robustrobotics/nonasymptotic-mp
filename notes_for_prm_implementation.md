# PRM Implementation (since ompl doesn't give us the level of control we need)

## Ingredients:
- Efficient and relatively accurate NN datastructure
- An alternate way to represent the roadmap grap
- An efficient implementation of a graph search algorithm.
- A way to check the validity of edges given our current representation of the hypercube environment.

## Interface Requirements:
- Similar interface to OMPL... except we only care about real vector spaces with the default Euclidean metric
- Can consume a sampler, a motion validity checker, and then can do queries on its own.
- Should be compatible with a (slow) exact nearest-neighbor library and a (faster) ANN library (like FLANN)

## General implementation guidelines from reading the OMPL:
* But we're not multithreading! For reasons related to Python + SuperCloud does not like applications
internally multithreading anyway.
* We are implementing simple PRM ONLY, so we are not including the random bouncing motions. 
  - all content for this is in growRoadmap(), which is sample n points and add them as milestones.
    - Interested in ConnectionStrategy (neighbor lookup) add nn->add() (adding to NN)
      - Adding to the NN lookup: they add if there is room in the index. But if there isn't, then
      - rebuild the index. For our purposes, it may be better to just do everything once we're done sampling.
  - we'll start with an explicit representation... if it's feasible with ann.
* Glancing at LazyPRM (OMPL)... it's the same, without checking edge validity up front. 
* Our implementation of sPRM:
  * Building roadmaps:
    1. Samples all points up front.
    2. Construct index (and pass in old matrix to help).
    3. Grow the networkit graph (using implemented motion validity checker).
  * Handling motion-planning queries:
    1. Do the NN lookup to see if the start and goal are in the graph
    2. Connect start and goal to graph.
    3. Run A star
    4. Delete the start and goal from the graph.
  