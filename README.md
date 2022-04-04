# scoby
Spectra from Catalogs of OB stars

This code is for making synthesized spectra and radiation field maps based on observed catalogs of early-type stars.
If you have a stellar catalog with OB stars with known (or estimated) spectral types in it, this code can make a map of G0 (the far-ultraviolet radiation field) or ionizing flux throughout the region around these stars.
It can also estimate the wind or radiation outputs from the entire cluster.
The estimates rely on the [POWR](https://www.astro.physik.uni-potsdam.de/~wrh/PoWR/powrgrid1.php) models and some other stellar models.
It is similar to the cluster property synthesis done in the [Starburst99](https://www.stsci.edu/science/starburst99/docs/default.htm) code, but this works from an observed catalog rather than a synthetic distribution of stars.

To use this code, you will have to download some of the POWR models.

I have been developing this software since 2019 but it was buried in [a sea of other code](https://github.com/ramseykarim/feedback) for my FEEDBACK project thesis work.
I am now officially splitting it off as its own package.

Feel free to adapt this code to your own needs or contact me for help!
