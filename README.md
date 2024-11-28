# nonasymptotic-mp

The accompanying codebase for the publication "Towards Practical Finite Sample Bounds for Motion Planning in TAMP," by
Seiji A Shaw, Aidan Curtis, Leslie Pack Kaelbling, Tomás Lozano-Pérez, and Nicholas Roy.

## Quickstart

To run the Task and Motion Planning experiments, see
[the TAMP README.md](/exps/tamp_environment/README.md).

To view the implementations of PRM and numerical computations of the bound, see
[the nonasymptotic README.md](/nonasymptotic/README.md).

For examples of how the environments/PRMs are constructed, see 
[vis_narrow_passages.ipynb](/notebooks/vis_narrow_passages.ipynb).

For examples for how to use the numerical bound computations, see
[numerical_bound_computations.ipynb](/notebooks/numerical_bound_computations.ipynb).



## Installation

This code was developed and verified to work for Python 3.8. 

The minimal installation can be done by cloning down the repository and its submodules,
and then installing all dependencies in `requirements.txt`:

```shell
git clone --recursive git@github.com:robustrobotics/nonasymptotic-mp.git
cd nonasymptotic-mp
pip install -r requirements.txt
```

The minimal installation uses [pynndescent](https://pynndescent.readthedocs.io/en/stable/) 
to construct an approximate K-nearest neighbors graph that forms the
PRM. While installable by pip, the KNN graph construction slows down significantly when the input set of points 
grows larger than ~1e5.

You can optionally install [kgraph](https://github.com/aaalgo/kgraph), a much faster ANN library. 
The installation procedure has components that must be built from source. 
Please refer to the [kgraph repo](https://github.com/aaalgo/kgraph) for an installation procedure.


  
## Citation

If this codebase was helpful to you, please consider citing our paper:
```text
@inproceedings{
    shaw2024towards,
    title={Towards Practical Finite Sample Bounds for Motion Planning in {TAMP}},
    author={Seiji A Shaw and Aidan Curtis and Leslie Pack Kaelbling and Tom{\'a}s Lozano-P{\'e}rez and Nicholas Roy},
    booktitle={The 16th International Workshop on the Algorithmic Foundations of Robotics},
    year={2024},
    url={https://openreview.net/forum?id=I4pLUVhpU6}
}
```