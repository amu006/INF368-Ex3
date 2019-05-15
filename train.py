# The imported generators expect to find training data in data/train
# and validation data in data/validation
from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.optimizers import SGD, Adam
from keras.backend import eval as Keval

import os

from create_model import create_base_network, in_dim, tripletize, std_triplet_loss, alt_triplet_loss
from generators import triplet_generator
import testing as T

import config as C

from am_plankton import set_trainable_layers

last = C.last

def save_name(i):
    return (os.path.join(C.model_dir,'epoch_'+str(i)+'.model'))

def log(s):
    with open(C.logfile, 'a') as f:
        print(s, file=f)
        
def avg(x):
    return sum(x)/len(x)        

# Use log to file
logger = CSVLogger(C.logfile, append=True, separator='\t')

def train_step(trainable_n=None, optimizer=Adam()):
    if not trainable_n is None:
        base_model = model.layers[-2]
        log(set_trainable_layers(base_model, n=trainable_n))
        log(compile_model(model, optimizer=optimizer))
    model.fit_generator(
        triplet_generator(C.batch_size, None, C.train_dir), 
        steps_per_epoch=C.steps_per_epoch, epochs=C.epochs_per_step,
        callbacks=[logger],
        validation_data=triplet_generator(C.batch_size, None, C.val_dir), validation_steps=100)
        
def compile_model(model, optimizer=Adam()):
    #model.compile(optimizer=SGD(lr=C.learn_rate, momentum=0.9),
    #             loss=std_triplet_loss())
    model.compile(optimizer=optimizer,
                 loss=std_triplet_loss())
    return 'Model compiled with lr={}'.format(Keval(optimizer.lr))        

if last==0:
    log('Creating base network from scratch.')
    if not os.path.exists(C.model_dir):
        os.makedirs(C.model_dir)
    base_model = create_base_network(in_dim)
    C.learn_rate = C.learn_rate_initial
else:
    log('Loading model:'+save_name(last))
    base_model = load_model(save_name(last))
    C.learn_rate = C.learn_rate_subsequent

optimizer = Adam(lr=C.learn_rate)
model = tripletize(base_model)
log(compile_model(model, optimizer=optimizer))

vs = T.get_vectors(base_model, C.val_dir)
cents = {}
for v in vs:
    cents[v] = T.centroid(vs[v])

for i in range(last+1, last+11):
    log('Starting step '+str(i)+'/'+str(last+10)+' lr='+str(C.learn_rate))
    if i==1: #first step: train top layer only
        train_step(trainable_n=1, optimizer=optimizer)
    elif i==2: #second step: unlock and train full model
        C.learn_rate = C.learn_rate_full
        optimizer = Adam(lr=C.learn_rate)
        train_step(trainable_n=999, optimizer=optimizer) #unlock all layers
    else:
        train_step(optimizer=optimizer)  
    C.learn_rate = C.learn_rate * C.lr_decay
    base_model.save(save_name(i))

    vs = T.get_vectors(base_model, C.val_dir)
    T.save_obj(vs, os.path.join(C.obj_dir, 'val_pred_'+str(i)))
    c = T.count_nearest_centroid(vs)
    log('Summarizing '+str(i))
    with open(os.path.join(C.log_dir, 'summarize.'+str(i)+'.log'), 'w') as sumfile:
        T.summarize(vs, outfile=sumfile)
    with open(os.path.join(C.log_dir, 'clusters.'+str(i)+'.log'), 'w') as cfile:
        T.confusion_counts(c, outfile=cfile)
    c_tmp = {}
    r_tmp = {}
    for v in vs:
        c_tmp[v] = T.centroid(vs[v])
        r_tmp[v] = T.radius(c_tmp[v], vs[v])
    c_rad = [round(100*r_tmp[v])/100 for v in vs]
    c_mv = [round(100*T.dist(c_tmp[v],cents[v]))/100 for v in vs]
    log('Centroid radius: '+str(c_rad))
    log('Centroid moved: '+str(c_mv))
    cents = c_tmp

    with open(C.logfile, 'a') as f:
        T.accuracy_counts(c, outfile=f)
    # todo: avg cluster radius, avg cluster distances
    log('Avg centr rad: %.2f move: %.2f' % (avg(c_rad), avg(c_mv)))
