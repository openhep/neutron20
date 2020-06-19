# neutron20

Studying generalized parton distributions of quarks in nucleon using 
model and neural network fits to JLab's deeply virtual Compton scattering data.


This is the code for the analysis done for the paper

   * Čuić, M., Kumerički, K., and Schäfer A.,  _Separation of Quark Flavors using DVCS Data_, to be finished soon

Jupyter/Python notebook ``jlab20-fits.ipynb``  above shows in detail how numerical results
and plots are obtained and provides some plots additional to what is in the paper.
This file is commented and should be readable immediately here on
the github. If you want to run it yourself, you need implementation of formulas
connecting form factors with cross-sections (Belitsky, Mueller et al. papers).
This can be obtained from the authors upon request (and will be separately
published).

Subdirectory ``data`` contains Fourier transforms of experimental data used.
File ``fits20.db`` is Python shelve database of values of observables for all experimental
points used, as calculated by each model from the paper.

If you find bugs, please, report them either through github "Issues" or directly to
authors via email (easily found at [arXiv](http://arXiv.org)).
