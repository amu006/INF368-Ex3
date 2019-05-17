# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:12:37 2019

@author: angus

Visualisations for exercise 3
"""

import os
import numpy as np
#from statistics import stdev
from copy import deepcopy
from keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
import plotly.graph_objs as go
import testing as T
import config as C

prop_cycle = plt.rcParams['axes.prop_cycle']
colours = prop_cycle.by_key()['color']

def Average(lst):  #average of a list
    return sum(lst) / len(lst) 

def stdev(lst):
    """ bug in the bloody statistics.stdev so do it myself """
    a = Average(lst)
    nlst = lst - a
    s = sum([x**2 for x in nlst])
    return np.sqrt(s/(len(lst)-1))

def read_clusters(ldir=None):
    """
    Reads cluster log files from the given directory.
    Returns a nested dictionary of arrays of length (num files)
    """
    if ldir is None:
        ldir = os.getcwd()
    cfiles = [f for f in os.listdir(ldir) if f[:9]=='clusters.']
    cidxs = [int(f.split('.')[1]) for f in cfiles]
    if not len(cidxs)==max(cidxs):
        raise Exception("Incomplete cluster log file set!")
    res = {}
    for i in range(len(cfiles)):
        with open(os.path.join(ldir,cfiles[i])) as cf:
            line = cf.readline()
            count = 1
            this_file = []
            while line:
                this_file.append(line.split())
                line = cf.readline()
                count += 1
        cc = [l[0] for l in this_file] #current classes
        #initialise dict first time round:
        if len(res.keys())==0:
            for j in range(len(cc)):
                res[cc[j]] = {}
                for k in range(len(cc)):
                    res[cc[j]][cc[k]] = np.zeros((len(cidxs)), dtype=int)
        #populate current file into res dict:
        for ci in range(len(this_file)): #file row
            for cj in range(len(this_file)): #file column
                res[this_file[ci][0]][this_file[cj][0]][cidxs[i] - 1] = \
                    this_file[ci][cj + 1]
    return res

def read_summaries(ldir=None):
    """
    Reads summary log files from the given directory
    Returns a dictionary with items:
        'r': array of radii of the current class cluster at each iteration
        'dist': dict of arrays of distances between this and other classes'
            centroids at each iteration, keys are other class names
        length (num files)
    """
    if ldir is None:
        ldir = os.getcwd()
    sfiles = [f for f in os.listdir(ldir) if f[:10]=='summarize.']
    sidxs = [int(f.split('.')[1]) for f in sfiles]
    if not len(sidxs)==max(sidxs):
        raise Exception("Incomplete summary log file set!")
    res = {}
    for i in range(len(sfiles)):
        with open(os.path.join(ldir,sfiles[i])) as sf:
            line = sf.readline()
            count = 1
            this_file = []
            while line:
                this_file.append(line.split())
                line = sf.readline()
                count += 1
        cc = [l[0] for l in this_file] #current classes
        #initialise dict first time round:
        if len(res.keys())==0:
            if not len(this_file)==len(cc):
                raise Exception("Incorrect num rows in file")
            for j in range(len(this_file)):
                res[this_file[j][0]] = {'dist': {}, 
                   'r': np.zeros((len(sidxs)), dtype=np.float32) }
                for k in range(len(cc)):
                    res[this_file[j][0]]['dist'][cc[k]] = \
                        np.zeros((len(sidxs)), dtype=np.float32)                        
                        
        #populate current file into res dict:
        for ci in range(len(this_file)): #file row
            res[this_file[ci][0]]['r'][sidxs[i] - 1] = np.float32(this_file[ci][1][2:])
            for cj in range(len(this_file)): #file column
                if ci==cj and not float(this_file[ci][cj + 2])==0:
                    raise Exception("Data is buggered: nonzero cluster distance diagonal")
                res[this_file[ci][0]]['dist'][this_file[cj][0]][sidxs[i] - 1] = \
                    this_file[ci][cj + 2]
    return res

def read_validation_history(obj_dir=C.obj_dir):
    """ load all pickled validation predictions """
    obj_names = [o.split('.')[-2] for o in os.listdir(obj_dir)]
    its = [int(name.split('_')[-1]) for name in obj_names] #iteration numbers
    vect_hist = {}
    for n, i in enumerate(its):
        vect_hist[int(i)] = dict_squeeze(T.load_obj(os.path.join(obj_dir, obj_names[n])))
    return vect_hist

def confusion_matrix(clusters, iteration):
    """
    Extracts a confusion matrix from cluster dictionary
    for the given iteration
    """
    
    nc = len(clusters)
    #ni = len(clusters[list(clusters)[0]][list(clusters)[0]])
    cmat = np.zeros((nc, nc), dtype=np.int32)
    classes = [c for c in clusters]
    for i in range(len(classes)):
        for j in range(len(classes)):
            cmat[i,j] = clusters[classes[i]][classes[j]][iteration]
    return cmat, classes

def dict_squeeze(vectors):
    """ 
    Squeezes the lists in a vector dict (as returned by get_vectors)
    into arrays for further analysis.
    
    Is idempotent
    """
    for c in vectors:
        if type(vectors[c]) is list:
            vectors[c] = np.squeeze(vectors[c])
    return vectors

def dict_unsqueeze(vectors):
    """
    Unsqueezes the arrays in a vector dict (result of dict_squeeze)
    into lists of vectors (as returned by get_vectors)
    
    Is idempotent
    """
    for c in vectors:
        if not type(vectors[c]) is list:
            vectors[c] = [np.array(vectors[c][r,:]) for 
                   r in range(vectors[c].shape[0])]
    return vectors
        
def collect_class_vs(vectors, classes=None):
    """ Collects vectors in given classes list together for analysis """
    if classes is None:
        classes = [c for c in vectors]
    X = np.squeeze(np.concatenate([vectors[c] for c in classes]))
    #X is a master list of all vectors of these classes
    y = np.concatenate([[c]*len(vectors[c]) for c in classes])
    return X, y
    
def standardise_vectors(vs, reference_classes=None):
    """
    Standardises all vectors with respect to the 
    mean and std of vectors in reference_classes
    """
    if reference_classes is None:
        reference_classes = [c for c in vs]
    X, y = collect_class_vs(vs, reference_classes)
    X_mean = np.expand_dims(np.mean(X, axis=0), axis=0)
    X_std = np.expand_dims(np.std(X, axis=0), axis=0)
    #X -= X_mean
    #X /= X_std
    #now X is normalised array of vectors from reference_classes
    vs_n = {}
    for c in vs: #for all classes:
        vs_n[c] = (np.squeeze(vs[c]) - X_mean) / X_std
    return vs_n, X_mean, X_std

def svd_project(vs, reference_classes=None):
    """
    Projects all vectors onto the left singular vector basis
    (basis is computed on reference_classes)
    """
    if reference_classes is None:
        reference_classes = [c for c in vs]
    vs_n, X_mean, X_std = standardise_vectors(vs, reference_classes)
    X, y = collect_class_vs(vs_n, reference_classes)
    U, S, Vh = np.linalg.svd(X.T, full_matrices=False)
    #X.T has columns=data 64-vectors
    # U has columns = left singular vectors of data
    # S is vector of singular values
    # Vh has columns = coeffs of each left singular vector in original data vectors
    #Now transform class data into basis comprised by U:
    ##ws_n = {}
    ##for c in classes:
    ##    ws_n[c] = vs_n[c] @ U
    return {c: vs_n[c] @ U for c in vs_n}        
        

#plotting:
    
def plot_epoch_losses(textfile=None):
    """ Plot the training loss over epochs """
    if textfile is None:
        textfile = os.path.join(C.log_dir, 'training.txt')
    with open(textfile) as f:
        ll = f.readlines()
    loss_train = [float(l.split()[7]) for l in ll[1::2]]
    loss_val = [float(l.split()[10]) for l in ll[1::2]]
    epochs = [e for e in range(len(loss_train))]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(epochs, loss_train, label='Training loss')
    ax.plot(epochs, loss_val, label='Validation loss')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set(xlabel='Epoch')
    ax.set(ylabel='Loss')
    return fig

def plot_train_radii(summ):
    """
    Plots the radii of each cluster during training
    """    
    fig, ax = plt.subplots(1)
    for c in summ:
        iteration = [i + 1 for i in range(len(summ[c]['r']))]
        ax.plot(iteration, summ[c]['r'], label=c)
    ax.plot(iteration, np.mean(np.array(
            np.transpose([(summ[c]['r']) for c in summ])), axis=1), 
            linewidth=5, linestyle='dashed', label='MEAN')
    ax.set(xticks=iteration)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set(xlabel='Iteration')
    ax.set(ylabel='radius')
    return fig

def plot_train_radii_separation(obj_dir=C.obj_dir, std=False):
    """ 
    Plots the average cluster radius and average distance to other clusters
    during training
    """
    vh = read_validation_history(obj_dir) #is squeezed
    centroids = {}
    radii = {}
    distances = {}
    av_dist = {}
    dist_all = {}
    for i in vh:
        #unsqueeze as we go:
        vh[i] = dict_unsqueeze(vh[i])
        centroids[i] = {}
        radii[i] = {}
        distances[i] = {}
        dist_all[i] = []
        for c in vh[i]:
            centroids[i][c] = T.centroid(vh[i][c])
            radii[i][c] = T.radius(centroids[i][c], vh[i][c])
        av_dist[i] = {}    
        for c in vh[i]:
            distances[i][c] = {}
            for c2 in vh[i]:
                distances[i][c][c2] = T.dist(centroids[i][c], centroids[i][c2])
                dist_all[i].append(distances[i][c][c2])
            av_dist[i][c] = Average([distances[i][c][c2] for c2 in distances[i]])
    its = [i for i in vh]
    its.sort()
    rads = [Average([radii[i][c] for c in radii[i]]) for i in its]
    rads_std = [stdev([radii[i][c] for c in radii[i]]) for i in its]
    dists = [Average([av_dist[i][c] for c in av_dist[i]]) for i in its]
    
    dists_std = [stdev(dist_all[i]) for i in its]
    fig, ax = plt.subplots(1)
    ax.errorbar(its, rads, yerr=rads_std, linewidth=3, linestyle='dashed', 
                label='Radii', elinewidth=1, capsize=10)
    ax.errorbar(its, dists, yerr=dists_std, linewidth=3, 
                label='Average separation', elinewidth=1, capsize=10)
    ax.legend()
    ax.set(xlabel='Iteration')
    ax.set(ylabel='radius')  
    fig.suptitle('Average class radii and separation during training', fontsize=16)
    return fig
    
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.show()
    return fig

def plot_class_vectors_scatter(vectors, classes, dims=[0,1,2]):
    """
    Creates a 3D scatterplot of the given classes in dictionary of vectors. 
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    #colours = ['blue', 'red', 'green']
    for i, c in enumerate(classes):
        #if type(vectors[c]) is list:
        #    tmp = np.squeeze(vectors[c])
        #else:
        tmp = vectors[c]
        ax.scatter(tmp[:, dims[0]], 
                   tmp[:, dims[1]], 
                   tmp[:, dims[2]], color=colours[i%len(colours)])
    return fig

def plot_class_vectors_plotly(vectors, classes, dims=[0,1,2],
                              filename='mesh3d_sample'):
    """
    Creates a 3D plotly plot of the given classes in dictionary of vectors
    """
    scatter = []
    cluster = []
    for i, c in enumerate(classes):
        scatter.append(dict(
            mode = "markers",
            name = c,
            type = "scatter3d",    
            x = vectors[c][:,dims[0]], 
            y = vectors[c][:,dims[1]], 
            z = vectors[c][:,dims[2]],
            marker = dict( size=2, color=colours[i%len(colours)])
            ))
        cluster.append(dict(
            alphahull = 5,
            name = c, #"Cluster {}".format(i),
            opacity = .1,
            type = "mesh3d",    
            x = vectors[c][:,dims[0]], 
            y = vectors[c][:,dims[1]], 
            z = vectors[c][:,dims[2]],
            color=colours[i%len(colours)], showscale = True
            ))
    layout = dict(
            title = 'Interactive Cluster Shapes in 3D, dims={}'.format(dims),
            scene = dict(
                    xaxis = dict( zeroline=True ),
                    yaxis = dict( zeroline=True ),
                    zaxis = dict( zeroline=True ),
                    )
            )
    fig = dict( data=[*scatter, *cluster], layout=layout )
    # Use py.iplot() for IPython notebook
    plot(fig, filename)
    return

def plotly_animate(validation_history, classes=None, mesh=True, dims=[0,1,2],
                   ax_lims=1.):
    """ Create a kick-ass animation (3D) of cluster evolution during training """
    vh = validation_history
    its = list(vh)
    its.sort()
    if classes is None:
        classes = list(vh[its[0]])
    
    #at step i:
    #class c is:
    #vh[i][c] - np array (100x64) - will plot the 1st 3 dims as demo
    axl = ax_lims
    
    scatter = {}
    cluster = {}
    for e in its:
        scatter[e] = [
                dict(
                mode = "markers",
                name = c,
                type = "scatter3d",    
                x = vh[e][c][:,dims[0]], 
                y = vh[e][c][:,dims[1]], 
                z = vh[e][c][:,dims[2]],
                marker = dict( size=2, color=colours[i%len(colours)])
                ) for i, c in enumerate(classes)
                ]
        cluster[e] = [
                dict(
                alphahull = 5,
                name = c, #"Cluster {}".format(i),
                opacity = .1,
                type = "mesh3d",    
                x = vh[e][c][:,dims[0]], 
                y = vh[e][c][:,dims[1]], 
                z = vh[e][c][:,dims[2]],
                color=colours[i%len(colours)], showscale = True
                ) for i, c in enumerate(classes)
                ]
    #starting data:     
    if mesh:
        data = [*scatter[1], *cluster[1]]
    else:
        data = scatter[1]
    #list of dicts of data items to update at each step  
    if mesh:
        frames=[dict(data = [*scatter[e], *cluster[e]],
                     name = 'frame{}'.format(e)       
                     ) for e in its]
    else:
        frames=[dict(data = scatter[e],
                     name = 'frame{}'.format(e)       
                     ) for e in its]
                
    sliders=[dict(steps= [dict(method= 'animate',#Sets the Plotly method to be called when the
                                                    #slider value is changed.
                               args= [['frame{}'.format(e)],#Sets the arguments values to be passed to 
                                                                  #the Plotly method set in method on slide
                                      dict(mode= 'immediate',
                                           frame= dict(duration=300, redraw=False),
                                           transition=dict(duration=300, easing='cubic-in-out')
                                           )
                                        ],
                                label='{}'.format(e)
                                 ) for e in its], 
                    transition= dict(duration= 300, easing='cubic-in-out'),
                    currentvalue=dict(font=dict(size=12), 
                                      prefix='Step: ', 
                                      visible=True, 
                                      xanchor= 'center'
                                     ),
                    active=0,
                    len=1.0)#slider length)
               ]
        
    layout = dict(
                title = 'Interactive Cluster Shapes in 3D, dims={}'.format(dims),
                scene = dict(
                        xaxis = dict(range=[axl, -axl], zeroline=True ),
                        yaxis = dict(range=[axl, -axl], zeroline=True ),
                        zaxis = dict(range=[-axl, axl], zeroline=True ),
                        camera = dict(eye=dict(x=1.2, y=1.2, z=1.2)),
                        aspectratio = dict(x=1, y=1, z=1),
                        ),
                sliders=sliders
                )
                        
    fig=dict(data=data, layout=layout, frames=frames)
    plot(fig, validate=False)
    #plot(fig, validate=False)
    return


################################################################s
#demos:
    
clust = read_clusters()
summ = read_summaries()

#Plot the training progression of cluster radii:
plot_train_radii(summ)

#Plot the average cluster radius compared to the average cluster separation:
plot_train_radii_separation(C.obj_dir)

#Plot the training loss over epochs:
plot_epoch_losses()        

#Plot a confusion matrix
i = 20
cmat, classes = confusion_matrix(clusters, i-1)
plot_confusion_matrix(cmat,
                          classes,
                          title='Confusion matrix for {}th iteration'.format(i),
                          cmap=None,
                          normalize=True)




#Load all models and pickle their validation predictions
model_names = os.listdir(C.model_dir)
for model_name in model_names:
    oname = model_name.split('.')[0]+'_val_pred'
    T.save_model_predictions(os.path.join(C.model_dir,model_name), tdir=C.val_dir, 
                       ofile=os.path.join(C.obj_dir,oname))


    

vs = dict_squeeze(T.load_obj('obj/111/val_pred_11'))
classes = [c for c in vs]        
#plot some classes' first 3 dimensions    
plot_class_vectors_scatter(vs, [classes[i] for i in range(16)], 
                                  dims=[0,1,2])    

#plot some classes along 3 principal axes of overall data
ws_n = svd_project(vs)
plot_class_vectors_scatter(ws_n, [classes[i] for i in range(16)])
plot_class_vectors_plotly(ws_n, [classes[i] for i in range(10)])

#Make an animation of validation predictions at different training epochs:
#plotly.plotly.create_animations()

#init_notebook_mode(connected=True)

vh = read_validation_history(obj_dir)
classes = list(vh[1])[:6]
plotly_animate(vh, classes=classes)

#Make animation of svd of training val preds:
vh = read_validation_history(obj_dir)
wh = {i: svd_project(vh[i]) for i in vh}
classes = list(wh[1])[:6]
plotly_animate(wh, classes=classes, ax_lims=6)
