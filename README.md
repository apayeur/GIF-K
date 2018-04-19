Fitting a GIF neural model with voltage-dependent potassium current to data

This code builds upon the code found in https://github.com/pozzorin/GIFFittingToolbox  It implements a GIF (Generalized Integrate-and-Fire, a close cousin of GLM and SRM) neuron model with subthreshold voltage-dependent potassium current, and fits its parameters to experimental data. The potassium current appears as an added term in the total current of the basic GIF model.

The main methods used in this code are a multilinear regression for the subthreshold parameters and maximum likelihood for matching the model spike initiation parameters to spiking data.

The core code has been created by Christian Pozzorini, Skander Mensi and Richard Naud from EPFL. The fitting protocol has been designed by Richard Naud, C. Pozzorini and S. Mensi under the supervision of Wulfram Gerstner.

You can readily tell apart my contribution by comparing my repository with the one listed above. If you ever use the present repository, please refer both to this repository and the GIFFittingToolbox.

References: [1] Pozzorini, C., Mensi, S., Hagens, O., Naud, R., Koch, C., & Gerstner, W. (2015). Automated high-throughput characterization of single neurons by means of simplified spiking models. PLoS computational biology, 11(6), e1004275. [2] Mensi, S., Naud, R., Pozzorini, C., Avermann, M., Petersen, C. C., & Gerstner, W. (2012). Parameter extraction and classification of three cortical neuron types reveals two distinct adaptation mechanisms. Journal of neurophysiology, 107(6), 1756-1775. [3] Frazier, C. J., Serrano, J. R., George, E. G., Yu, X., Viswanathan, A., Perez-Reyes, E., & Jones, S. W. (2001). Gating kinetics of the α1I T-type calcium channel. The Journal of general physiology, 118(5), 457-470.
