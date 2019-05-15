
train_dir = 'train'
val_dir   = 'validate'
test_dir  = 'test'

model_dir = 'models'
obj_dir = 'obj'
log_dir = 'log'
logfile    = 'train.log'  #this is in base directory, so name it to reflect log_dir


batch_size = 20
steps_per_epoch = 600  #1000
epochs_per_step = 5  #formely iterations

last       = 1 #0 for new model. If not 0, adjust learn_rate_subsequent

learn_rate = 0.01  
lr_decay   = 0.9

#for Adam:
learn_rate_initial = 1E-3
learn_rate_subsequent = 1E-5  #for resuming training, match this to decayed lr from logfile!


