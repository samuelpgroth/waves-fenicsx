# waves-fenicsx
Repository containing python scripts to solve linear and non-linear acoustic wave equations using the open-source finite-element software [FEniCS-X](https://github.com/FEniCS/dolfinx). Of particular interest are simulations pertaining to scenarios arising in high-intensity focused ultrasound applications. Such scenarios are typically modelled using the Westervelt equation, requiring time-domain solvers. For a time-harmonic source, Westervelt can be recast into a series of Helmholtz equations for subsequent harmonics. In this repo, we provide solvers for both settings.

## Using docker
To run FEniCS-X, we recommend using docker. The docker image can be obtained by executing <br>
```docker pull dolfinx/dolfinx```

To start the docker image, a command such as this should work:<br>
```docker run -it --shm-size=512m -v "$PWD":/root/fenics dolfinx/dolfinx:latest```
