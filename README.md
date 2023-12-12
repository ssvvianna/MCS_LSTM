# MCS_LSTM
Accidental gas releases from pipelines and storage units in Floating Production and Offloading (FPSO) vessels are significant safety concerns, necessitating real-time leakage prediction tools for prompt decision-making and efficient safety procedures. In this study, we introduce a predictive model based on Long Short-Term Memory (LSTM) networks coupled with a Bayesian approximation using Monte Carlo dropout to infer CH₄ leakages in FPSOs while addressing the uncertainty associated with model predictions.

Overview

Problem Statement
Accurate and timely prediction of CH₄ leakages in FPSO vessels is crucial for ensuring the safety of operations and personnel. Traditional methods often fall short in providing real-time insights and accounting for uncertainties inherent in the complex FPSO environment.

Approach
We leveraged LSTM networks, known for their ability to capture temporal dependencies, and integrated a Bayesian approximation using Monte Carlo dropout to enhance uncertainty quantification in our model.

Data

Source
The training and testing database utilized in this study were derived from 3D Computational Fluid Dynamics simulations. The dataset encompassed eight FPSO modules, three CH₄ discharge rates, three wind velocities, and eight wind directions. Additionally, a non-leakage dataset was included for comprehensive model training.

Input Variables
The model incorporated CH₄ concentrations from ninety-six sensors and wind specifications as input variables, ensuring a holistic representation of the FPSO environment.

Noise Assessment
To evaluate the model's robustness, we introduced noise to the data and assessed its impact on performance and uncertainty quantification.

Results

Our model demonstrated robust predictive capabilities, achieving a precision of 93.9% with noise-free data. Even in the presence of noise, the model maintained a high precision of 91.9%, showcasing its effectiveness in predicting leaks using readily available inputs while accounting for uncertainties in both the model and data.
