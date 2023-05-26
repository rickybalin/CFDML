# Data Driven Sub-Grid Stress (SGS) Modeling of Turbulent Flows

## To Do
- [ ] Set up DNS of a turbulent flat plate at $Re_\theta=1000$ with hexahedral elements and resolution of $d_x^+=15$ and $d_z^+=6$
  - [ ] Repartition case for Polaris
  - [ ] Run DNS for a few time steps to make sure setup is correct
- [ ] Extract training, validation and testing data from flat plate DNS
  - [ ] Run DNS saving to restart files the model inputs, output, and min-max scaling for the dataset
  - [ ] Repeat with a few filter widths scaling factors (3, 6, 12) and for three separate time steps
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


