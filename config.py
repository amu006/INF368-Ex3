
train_dir = 'train'
val_dir   = 'validate'
test_dir  = 'test'

batch_size = 20
steps_per_epoch = 600  #1000
epochs_per_step = 5  #formely iterations

logfile    = 'train.log'

last       = 1
learn_rate = 0.01
#for Adam:
learn_rate_initial = 1E-3
learn_rate_full = 1E-5
lr_decay   = 0.9
