# Data Driven Sub-Grid Stress (SGS) Modeling of Turbulent Flows

## To Do
- [x] Set up DNS of a turbulent flat plate at $Re_\theta=1000$ with hexahedral elements and resolution of $d_x^+=15$ and $d_z^+=6$
  - [x] Repartition case for Polaris
  - [x] Run DNS for a few time steps to make sure setup is correct
- [ ] Extract training, validation and testing data from flat plate DNS
  - [x] Run DNS saving to restart files the model inputs, output, and min-max scaling for the dataset
  - [x] Repeat with different width scaling factors (1, 3, 6, 12) for the same time step to generate training data
  - [ ] Repeat for different time steps to generate validation data
- [ ] Perform offline training of anisotropic SGS ANN model on flat plate DNS data
  - [ ] Train model on a single snapshot and a single filter width
  - [ ] Train model on a single snapshot and multiple filter widths
- [ ] Perform a priori tests of models trained on flat plate DNS data
  - [ ] Decide what flow to use, $Re_\theta=1000$ or also $Re_\theta=2000$?
  - [ ] Compare to current HIT trained model
  - [ ] Look at model performance at various heights off the wall-normal
  - [ ] Look at model performance at various streamwise locations (e.g., near STG inflow)
- [ ] Perform a posteriori tests
  - [ ] Decide what case to use


