# DIPE

train_source_model.py : train the source model
target_DIPE: train the target model with DIP. We replace the SGD with the newSGD to exploring the DIP.

init_source : train the target model only with cross entropy loss
init_sourceDIPE : train the target model with DIP and cross entropy loss
