# Models

Models will be saved here.

## Naming convention
Models will be saved in a directory that is named 
in accordence with this format: modelType_DD-Mon-YYYY_(hh:mm:ss).

All checkpoints and validation results will be saved in this directory. 
Test results will also be saved here. The required model for predict_model 
is also the name of this directory. predict_model will look for the best
checkpoint in this directory and load that checkpoint. 