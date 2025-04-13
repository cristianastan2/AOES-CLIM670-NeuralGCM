---
title: "Neural GCM"
teaching: 0
exercises: 0
questions:
- "What are neural general circulation models for weather and climate?"
objectives:
- "Learn about the artificial intelligence and machine learning models used for weather anc climate"
---

### What are artificial neural networks?
- Artificial neural networks (NNs) are mathematical models built to reflect the behavior of humain brain.
- If a sufficient amount of data are available, NNs can be trained to describe the evolution of non-linear processes.
- Due to the fundamentally application-agnostic character, no complete understanding of the underlying process is necessary.
- NNs have been used to:
    * post-process data from weather forecast models to optimise predictions:
      
      + [Krasnopolsky and Lin, 2012](https://doi.org/10.1155/2012/649450) 
        
      + [Rasp and Lerch, 2018](https://doi.org/10.1175/MWR-D-18-0187.1) 
        
    * radiation parameterisation in operational forecasts at ECMWF in the past:
      
      + [Chevallier et al., 1998](https://doi.org/10.1175/1520-0450(1998)037<1385:ANNAFA>2.0.CO;2) 
        
      + [Chevallier et al., 2000](https://doi.org/10.1002/qj.49712656318) 
        
      + [Krasnopolsky et al., 2005](https://doi.org/10.1175/MWR2923.1) 
        
    * parameterization of ocean physics:
      
      + [Krasnopolsky et al., 2002](https://doi.org/10.1016/S1463-5003(02)00010-0) 
        
      + [Tolman et al., 2005](https://doi.org/10.1016/j.ocemod.2003.12.008) 
        
    * parameterization of convection:
      
      + [Krasnopolsky et al., 2013](https://doi.org/10.1155/2013/485913) 
        
    * weather prediction:
      
      + [Dueben and Bauer, 2018](https://doi.org/10.5194/gmd-11-3999-2018)
        
      + [Scher and Messori, 2019](https://doi.org/10.5194/gmd-12-2797-2019) 
        
      + [Weyn et al. 2019](https://doi.org/10.1029/2019MS001705) 
         
      + [Weyn et al., 2020](https://doi.org/10.1029/2020MS002109) 
        
      + [Weyn et al., 2021](https://doi.org/10.1029/2021MS002502) 
        
      + [Keisler, 2022](https://arxiv.org/pdf/2202.07575): The first model to claim competitive performance with operational models on weather forecast time.
        
      + [Lang et al., 2024](https://arxiv.org/abs/2406.01465): ECMWF AIFS
      
      + [Kochkov et al. 2024](https://www.nature.com/articles/s41586-024-07744-y): NeuralGCM    

### NeuralGCM
- The material presented here is adopted from [Kochkov et al. 2024](https://www.nature.com/articles/s41586-024-07744-y)
- Is a fully differentiable **hybrid** GCM of Earth's atmosphere.

     - differentiable dynamical core for solving the discretized governing equations:
          + solves the dynamical equations of the atmosphere, describing large-scale fluid motion and thermodynamics under the influence of gravity and the Coriolis force.
          + uses a horizontal pseudo-spectral discretization and vertical sigma coordinates
          + seven prognostic variables: vorticity and divergence of horizontal wind, temperature, surface pressure, and three water species (specific humidity, and specific ice and liquid cloud water content)
          + differentiable dynamical core is implemented in JAX, a library for high-performance code in Python that supports automatic differentiation. 
     - a learned physics module that parameterizes physical processes with a neural network
          + predicts the effect of unresolved processes, such as cloud formation, radiative transport, precipitation and subgrid-scale dynamics, on the simulated fields using a neural network.
          + uses the single-column approach of GCMs
 
  ![image](https://github.com/user-attachments/assets/ebb11382-a035-4fbe-8ffb-53e7072ecebe)
  **a**: Overall model structure, showing how forcings F<sub>t</sub>, noise z<sub>t</sub> (for stochastic models) and inputs y<sub>t</sub> are encoded into the model state x<sub>t</sub>. The model state is fed into the dynamical core, and alongside forcings and noise into the learned physics module. This produces tendencies (rates of change) used by an implicit–explicit ordinary differential equation (ODE) solver to advance the state in time. The new model state x<sub>t</sub>+1 can then be fed back into another time step, or decoded into model predictions. **b**: The learned physics module, which feeds data for individual columns of the atmosphere into a neural network used to produce physics tendencies in that vertical column. Figure 1 in [Kochkov et al. 2024](https://www.nature.com/articles/s41586-024-07744-y)

- Is available at three horizontal resolutions with grid spacing of 2.8<sup>o</sup>, 1.4 <sup>o</sup>, and 0.7<sup>o</sup>.
- The inputs:
     * prognostic variables in the atmospheric column: total incident solar radiation, sea-ice concentration and SST
     * horizontal gradients of prognostic variables
     * all inpputs are standardized to have zero mean and unit variance
     * training data: ERA5
### NeuralGCM Results
#### Geostrophic balance
![image](https://github.com/user-attachments/assets/90bcfd63-1105-4235-bef0-eef779218ba7)
Vertical profiles of the extratropical intensity (averaged between latitude 30°–70° in both hemispheres) and over all forecasts initialized in 2020 of (a,d,g) geostrophic wind, (b,e,h) ageostrophic wind and (c,f,i) the ratio of the intensity of ageostrophic wind over geostrophic wind for ERA5 (black continuous line in all panels), (a,b,c) NeuralGCM-0.7°, (d,e,f) GraphCast and (g,h,i) ECMWF-HRES at lead times of 1 day, 5 days and 10 days. From [Kochkov et al. 2024](https://www.nature.com/articles/s41586-024-07744-y)

#### Precipitation minus evaporation
![image](https://github.com/user-attachments/assets/0a39ecb3-903d-48c7-88c9-372f8dab8845)
(a) Tropical (latitudes −20° to 20°) precipitation minus evaporation (P minus E) rate distribution, (b) Extratropical (latitudes 30° to 70° in both hemispheres) P minus E, (c) mean P minus E for 2020 ERA514 and (d) NeuralGCM-0.7° (calculated from the third day of forecasts and averaged over all forecasts initialized in 2020), (e) the bias between NeuralGCM-0.7° and ERA5, (f-g) Snapshot of daily precipitation minus evaporation for 2020-01-04 for (f) NeuralGCM-0.7° (forecast initialized on 2020-01-02) and (g) ERA5.

#### Tropical Cyclone densities and annual regional counts
![image](https://github.com/user-attachments/assets/9b62810d-8a63-42db-a738-aa016992f8ea)
(a) Tropical Cyclone (TC) density from ERA514 data spanning 1987–2020. (b) TC density from NeuralGCM-1.4° for 2020, generated using 34 different initial conditions all initialized in 2019. (c) Box plot depicting the annual number of TCs across different regions, based on ERA5 data (1987–2020), NeuralGCM-1.4° for 2020 (34 initial conditions), and orange markers show ERA5 for 2020. In the box plots, the red line represents the median; the box delineates the first to third quartiles; the whiskers extend to 1.5 times the interquartile range (Q1 − 1.5IQR and Q3 + 1.5IQR), and outliers are shown as individual dots. Each year is defined from January 19th to January 17th of the following year, aligning with data availability from X-SHiELD ([eXperimental System for High-resolution prediction on Earth-to-Local Domains developed at the Geophysical Fluid Dynamics Laboratory](https://doi.org/10.1029/2022GL099796)). For NeuralGCM simulations, the 3 initial conditions starting in January 2019 exclude data for January 17th, 2021, as these runs spanned only two years.

### Simulation of climate with NeuralGCM








