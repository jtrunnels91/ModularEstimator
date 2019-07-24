Modular Estimator (modest) is a package designed to help facilitate the implementation of a variety of estimation algorithms with a minimum amount of “boiler-plate” code. Modular estimator is designed around modularity, meaning that individual pieces of the estimation algorithm are built separately as much as possible. This allows for a high degree of flexibility in the configuration of the estimator, as well as for rigorous testing of sub-components in a controlled environment.

Some things the modest package offers include:

- A framework for designing estimators in a modular fashion with easily interchangeable sub-components
- A variety of built-in estimation algorithms, including an extended Kalman filter (EKF), a maximum likelihood (ML) estimator , and a joint probabilistic data association filter (JPDAF)
- The ability to easily compare performance between different estimation algorithms
- A framework for performing Monte Carlo simulations to evaluate the performance of a given estimation algorithm under controlled conditions

Please note that modest is currently still in the “alpha” development phase: this means that there are large portions of the code which are still somewhat undocumented/untested. Bug reports and suggestions for feature inclusion are welcomed!