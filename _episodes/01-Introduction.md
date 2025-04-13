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
- Due to the fundamentally application agnostic character, no complete understanding of the underlying process is necessary.
- NNs have been used to:
    * post-process data from weather forecast models to optimise predictions:
      ** ![https://doi.org/10.1155/2012/649450](Krasnopolsky and Lin, 2012)
      ** ![https://doi.org/10.1175/MWR-D-18-0187.1](Rasp and Lerch, 2018).
    * radiation parameterisation in operational forecasts at ECMWF in the past:
      ** ![https://doi.org/10.1175/1520-0450(1998)037<1385:ANNAFA>2.0.CO;2];![https://doi.org/10.1002/qj.49712656318](Chevallier et al., 1998);
      ** ![https://doi.org/10.1002/qj.49712656318](Chevallier et al., 2000)
      ** ![https://doi.org/10.1175/MWR2923.1](Krasnopolsky et al., 2005)
    * parameterization of ocean physics:
      ** ![https://doi.org/10.1016/S1463-5003(02)00010-0](Krasnopolsky et al., 2002)
      ** ![https://doi.org/10.1016/j.ocemod.2003.12.008](Tolman et al., 2005)
    * parameterization of convection:
      ** ![https://doi.org/10.1155/2013/485913](Krasnopolsky et al., 2013)
    * weather prediction:
      ** ![https://doi.org/10.5194/gmd-11-3999-2018](Dueben and Bauer, 2018)
      ** ![https://doi.org/10.5194/gmd-12-2797-2019](Scher and Messori, 2019)
      ** ![https://doi.org/10.1029/2019MS001705](Weyn et al. 2019) 
      ** ![https://doi.org/10.1029/2020MS002109](Weyn et al., 2020)
      ** ![ https://doi.org/10.1029/2021MS002502](Weyn et al., 2021)
      ** ![https://arxiv.org/pdf/2202.07575](Keisler, 2022): The first model to claim competitive performance with operational models on weather forecast time.
      ** ![https://arxiv.org/abs/2406.01465](Lang et al., 2024): ECMWF AIFS
      ** ![https://www.nature.com/articles/s41586-024-07744-y](Kochkov et al. 2024): NeuralGCM    

### NeuralGCM
- The material presented here is adopted from ![https://www.nature.com/articles/s41586-024-07744-y](Kochkov et al. 2024)
  
