# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:12:37 2019

@author: angus

Visualisations for exercise 3
"""

import os
import shutil
import numpy as np
#from statistics import stdev
from copy import deepcopy
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
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

def pickle_model_predictions(model_dir=C.model_dir, tdir=C.val_dir, obj_dir=C.obj_dir):
    """
    Saves pickle files to obj_dir, of predictions for the data in tdir, for each model 
    found in model_dir
    """
    model_names = os.listdir(model_dir)
    for model_name in model_names:
        print('Saving predictions from {}...'.format(model_name))
        oname = model_name.split('.')[0]+'_val_pred'
        T.save_model_predictions(os.path.join(C.model_dir,model_name), tdir=tdir, 
                           ofile=os.path.join(obj_dir,oname))
    return

def read_clusters(ldir=C.log_dir):
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

def read_summaries(ldir=C.log_dir):
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

def read_validation_history(obj_dir=C.obj_dir, prefix='val_pred'):
    """ load all pickled validation predictions """
    names = [o.split('.')[-2] for o in os.listdir(obj_dir)]
    obj_names =  [n for n in names if n[:8]==prefix]
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
    classes.sort()
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
    classes.sort()
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
    reference_classes.sort()
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

def svd_project(vs, reference_classes=None, return_items=False, from_U=False):
    """
    Projects all vectors onto the left singular vector basis
    (basis is computed on reference_classes)

    If return_items list is populated with any of the following names:
        'U', 'S', 'Vh', 'X_mean', 'X_std', 
        then these are also returned in a dictionary.
        (Note. Eval behaved strangely. So I just return the lot.)

    If from_U is specified (as an orthogonal transformation matrix),
    the SVD is bypassed and  the standardised data is simply transformed by U.
    """
    if reference_classes is None:
        reference_classes = [c for c in vs]
    reference_classes.sort()
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
    if from_U:
        res = {c: vs_n[c] @ from_U for c in vs_n}
    else:
        res = {c: vs_n[c] @ U for c in vs_n}        
    if return_items:
        return_dict = {'U': U, 'X_mean': X_mean, 'X_std': X_std}
        return res, return_dict
    else:
        return res
        

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
                          normalize=True,
                          figsize=(12,8)):
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

    fig = plt.figure(figsize=figsize)
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
    classes.sort()
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
                   tmp[:, dims[2]], color=colours[i%len(colours)], label=c)
    ax.set(xlabel='dim'+str(dims[0]), ylabel='dim'+str(dims[1]), zlabel='dim'+str(dims[2]))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig

def plot_class_vectors_plotly(vectors, classes, dims=[0,1,2],
                              filename='mesh3d_sample', notebook=False):
    """
    Creates a 3D plotly plot of the given classes in dictionary of vectors
    """
    classes.sort()
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
            color=colours[i%len(colours)], showscale = False
            ))
    layout = dict(
            title = 'Interactive Cluster Shapes in 3D, dims={}'.format(dims),
            scene = dict(
                    xaxis = dict( zeroline=True ),
                    yaxis = dict( zeroline=True ),
                    zaxis = dict( zeroline=True ),
                    ),
            legend = dict(itemsizing='constant'),
            )
    fig = dict( data=[*scatter, *cluster], layout=layout )
    # Use py.iplot() for IPython notebook
    if notebook:
        iplot(fig, filename)
    else:
        plot(fig, filename)
    return

def plotly_animate(validation_history, classes=None, mesh=True, dims=[0,1,2],
                   ax_lims=1., notebook=False):
    """ Create a kick-ass animation (3D) of cluster evolution during training """
    vh = validation_history
    its = list(vh)
    its.sort()
    if classes is None:
        classes = list(vh[its[0]])
    classes.sort()
    
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
                sliders=sliders,
                legend = dict(itemsizing='constant'),
                )
                        
    fig=dict(data=data, layout=layout, frames=frames)
    if notebook:
        iplot(fig, validate=False)
    else:
        plot(fig, validate=False)
    return

def plotly_animate_spheres(validation_history, classes=None, dims=[0,1],
                   ax_lims=1., notebook=False):
    """ 
    Create a kick-ass animation (3D) of cluster evolution during training, representing
    the clusters as spheres for clarity and speed   
    """
    vh = validation_history
    its = list(vh)
    its.sort()
    if classes is None:
        classes = list(vh[its[0]])
    classes.sort()
    #at step i:
    #class c is:
    #vh[i][c] - np array (100x64) - will plot the 1st 3 dims as demo
    axl = ax_lims

    #calculate centroids and radii:
    cent = {}
    rads = {}

    circles = {}
    for e in its:
        cent[e] = {c: T.centroid(dict_unsqueeze(vh[e])[c]) for c in classes}
        rads[e] = {c: T.radius(cent[e][c], dict_unsqueeze(vh[e])[c]) for c in classes}
        circles[e] = [
                dict(
                text=c,
                name=c,
                mode='markers',
                x=[cent[e][c][dims[0]]],
                y=[cent[e][c][dims[1]]],
                marker = dict(color=colours[i%len(colours)],
                            size=rads[e][c]*100,
                            ),
                ) for i, c in enumerate(classes)
                ]
    #starting data:     
    data = circles[1]
    #list of dicts of data items to update at each step  
    frames=[dict(data = circles[e],
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
                title = 'Interactive Cluster Shapes, dims={}'.format(dims),
                xaxis = dict(range=[-axl, axl], zeroline=True),
                yaxis = dict(range=[-axl, axl], zeroline=True),
                sliders=sliders,
                legend = dict(itemsizing='constant'),
                updatemenus=[{
                            'buttons': [
                                {
                                    'args': [None, {'frame': {'duration': 500, 'redraw': False},
                                            'fromcurrent': True, 'transition': transition}],
                                    'label': 'Play',
                                    'method': 'animate'
                                },
                                {
                                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                                    'transition': {'duration': 0}}],
                                    'label': 'Pause',
                                    'method': 'animate'
                                }
                            ],
                            'direction': 'left',
                            'pad': {'r': 10, 't': 87},
                            'showactive': False,
                            'type': 'buttons',
                            'x': 0.1,
                            'xanchor': 'right',
                            'y': 0,
                            'yanchor': 'top'
                        }]
                )
                        
    fig=dict(data=data, layout=layout, frames=frames)
    if notebook:
        iplot(fig, validate=False)
    else:
        plot(fig, validate=False)
    return

def plot_spheres(x, y, z, c, r):
    """ 
    Plots spheres representing the clusters in 3D 

    Inputs:
        x = 1D array or list of x centre values
        y = 1D array or list of y centre values
        z = 1D array or list of z centre values
        c = 1D array or list of colour values
        r = 1D array or list of radius values
    """
    def drawSphere(xCenter, yCenter, zCenter, r):
        #draw sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x=np.cos(u)*np.sin(v)
        y=np.sin(u)*np.sin(v)
        z=np.cos(v)
        # shift and scale sphere
        x = r*x + xCenter
        y = r*y + yCenter
        z = r*z + zCenter
        return (x,y,z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # draw a sphere for each data point
    for (xi,yi,zi,ri) in zip(x,y,z,r):
        (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
        ax.plot_wireframe(xs, ys, zs, color="r")

    return fig

def plotly_sphere(x0, y0, z0, r, c, name, n=25,
        colorscale='Jet', cmin=1, cmax=27):
    #todo: modify to spheroid
    if type(r) is float:
        r = [r,r,r]
    theta = np.linspace(0,2*np.pi,n)
    phi = np.linspace(0,np.pi,n)
    x = x0 + r[0]*np.outer(np.cos(theta),np.sin(phi))
    y = y0 + r[1]*np.outer(np.sin(theta),np.sin(phi))
    z = z0 + r[2]*np.outer(np.ones(n),np.cos(phi))  # note this is 2d now

    data = go.Surface(  name=name,
                        text=name,
                        x=x,
                        y=y,
                        z=z,
                        surfacecolor=c*np.ones(x.shape),
                        colorscale=colorscale,
                        cmin=cmin, 
                        cmax=cmax,
                        )
    return data

def plotly_spheres(validation_history, classes, dims=[0,1,2],
                   ax_lims=1., notebook=False, n=25):
    vh = validation_history
    its = list(vh)
    its.sort()
    if classes is None:
        classes = list(vh[its[0]])
    classes.sort()
    #at step i:
    #class c is:
    #vh[i][c] - np array (100x64) - will plot the 1st 3 dims as demo
    axl = ax_lims

    #calculate centroids and radii:
    cent = {}
    rads = {}

    circles = {}
    for e in its:
        cent[e] = {c: T.centroid(dict_unsqueeze(vh[e])[c]) for c in classes}
        rads[e] = {c: T.radius_nd(cent[e][c], dict_unsqueeze(vh[e])[c]) for c in classes}
        circles[e] = [
                plotly_sphere([cent[e][c][dims[0]]],
                               [cent[e][c][dims[1]]],
                                [cent[e][c][dims[2]]],
                                r=rads[e][c], 
                                c=i, #colours[i%len(colours)],
                                name=c,
                                colorscale='Jet',
                                cmin=1,
                                cmax=len(classes)+1,
                                n=n) for i, c in enumerate(classes)
                ]
    #starting data:     
    data = circles[1]
    #list of dicts of data items to update at each step  
    frames=[dict(data = circles[e],
                     name = 'frame{}'.format(e)       
                     ) for e in its]
    transition = dict(duration=300, easing='cubic-in-out')
    sliders=[dict(steps= [dict(method= 'animate',#Sets the Plotly method to be called when the
                                                    #slider value is changed.
                               args= [['frame{}'.format(e)],#Sets the arguments values to be passed to 
                                                                  #the Plotly method set in method on slide
                                      dict(mode= 'immediate',
                                           frame= dict(duration=500, redraw=False),
                                           transition={'duration': 300},
                                           )
                                        ],
                                label='{}'.format(e)
                                 ) for e in its], 
                    transition= transition,
                    currentvalue=dict(font=dict(size=12), 
                                      prefix='Step: ', 
                                      visible=True, 
                                      xanchor= 'center'
                                     ),
                    visible=True,
                    active=0,
                    pad={'b': 10, 't': 30},
                    x=0.1,
                    y=0,
                    len=1.0)#slider length)
               ]
        
    layout = go.Layout(
                title = 'Interactive 3D Cluster Shapes, dims={}'.format(dims),
                scene = dict(
                        xaxis = dict(range=[-axl, axl], zeroline=True),
                        yaxis = dict(range=[-axl, axl], zeroline=True),
                        zaxis = dict(range=[-axl, axl], zeroline=True),
                        #camera = dict(eye=dict(x=axl, y=axl, z=axl)),
                        aspectratio = dict(x=1, y=1, z=1),
                        ),
                sliders=sliders,
                legend = dict(itemsizing='constant'),
                updatemenus=[{
                            'buttons': [
                                {
                                    'args': [None, {'frame': {'duration': 500, 'redraw': False},
                                            'fromcurrent': True, 'transition': transition}],
                                    'label': 'Play',
                                    'method': 'animate'
                                },
                                {
                                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                                    'transition': {'duration': 0}}],
                                    'label': 'Pause',
                                    'method': 'animate'
                                }
                            ],
                            'direction': 'left',
                            'pad': {'r': 10, 't': 87},
                            'showactive': False,
                            'type': 'buttons',
                            'x': 0.1,
                            'xanchor': 'right',
                            'y': 0,
                            'yanchor': 'top'
                        }]
                )
                        
    fig=dict(data=data, layout=layout, frames=frames)
    if notebook:
        iplot(fig, validate=False)
    else:
        plot(fig, validate=False)
    return

def animate_and_save(plot_fn, out_file, arg_list, kwarg_list, fps=5.0):
    """
    Creates an animation of plots at all training steps
    and saves to a movie file
    
    Inputs: 
        plot_fn: function returning the pltos to be animated
        out_file: movie output filename
        arg_list: list of argument lists for plot_fn for each frame
        kwarg_list: list of kwarg dictionaries for plot_fn for each frame
        fps: frames per sec of the video desired
    
    with help from:
    https://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
    """
    os.makedirs('tmp', exist_ok=True)
    fnames = []
    for i in range(len(kwarg_list)):
        fig = plot_fn(*arg_list[i], **kwarg_list[i])
        fname = 'tmp/anim_{}.png'.format(i)
        fig.savefig(fname)
        plt.close(fig)
        fnames.append(fname)
    if os.path.exists(out_file):
        os.remove(out_file)
    frame = cv2.imread(fnames[0])
    #cv2.imshow('video',frame)
    height, width, channels = frame.shape
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') # Be sure to use lower case
    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    for fname in fnames:
        frame = cv2.imread(fname)
        out.write(frame) # Write out frame to video
        #cv2.imshow('video',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
    print("Output video {} written".format(out_file))
    shutil.rmtree('tmp')
    return
    

def animate_and_save_confusion(clust, 
                               out_file='notebooks/images/conf_anim.mp4',
                               fps=5.0):
    """
    Saves an animation of the confusion matrix through training """
    classes = list(clust)
    classes.sort()
    N = len(clust[classes[0]][classes[0]])
    arg_list = []
    kwarg_list = []
    for i in range(N):
        #save the confusion plot
        cmat, classes = confusion_matrix(clust, i)
        fig = plot_confusion_matrix
        arg_list.append([cmat, classes])
        kwarg_list.append(dict(title='Confusion matrix for {}th iteration'.format(i+1),
                                 cmap=None,
                                 normalize=True)
                                )
    animate_and_save(plot_confusion_matrix, out_file, arg_list, kwarg_list, fps)
    return

def main():
    pass

################################################################s
#demos:
    
if __name__ == "__main__":
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
    cmat, classes = confusion_matrix(clust, i-1)
    plot_confusion_matrix(cm=cmat,
                          target_names=classes,
                          title='Confusion matrix for {}th iteration'.format(i),
                          cmap=None,
                          normalize=True)
    
    #make animation of confusion matrixes through training process
    animate_and_save_confusion(clust)
    
    
    #Load all models and pickle their validation predictions
    #pickle_model_predictions()
    
    #plot some classes' first 3 dimensions  
    vs = dict_squeeze(T.load_obj('obj/111/val_pred_11'))
    classes = [c for c in vs]    
    classes.sort()    
    plot_class_vectors_scatter(vs, [classes[i] for i in range(16)], 
                                      dims=[0,1,2])    
    
    #plot some classes along 3 principal axes of overall data
    ws_n = svd_project(vs)
    plot_class_vectors_scatter(ws_n, [classes[i] for i in range(16)])
    plot_class_vectors_plotly(ws_n, [classes[i] for i in range(10)])
    
    #Make an animation of validation predictions at different training epochs:
    #plotly.plotly.create_animations()
    
    #init_notebook_mode(connected=True)
    
    vh = read_validation_history()
    classes = list(vh[1])[:6]
    plotly_animate(vh, classes=classes)
    
    #Make animation of svd of training val preds:
    vh = read_validation_history()
    wh = {i: svd_project(vh[i]) for i in vh}
    classes = list(wh[1])[:6]
    plotly_animate(wh, classes=classes, ax_lims=6)
    plotly_animate_circles(wh, classes=classes, ax_lims=6)

    #Train alternative classifiers on the validation data, test on test data.
    #prepare data:
    from sklearn.svm import SVC, LinearSVC
    from sklearn import tree
    from sklearn import cluster
    from sklearn.preprocessing import LabelBinarizer, LabelEncoder
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix as conf_mat
    from viz import *
    import config as C
    import testing as T

    classes = list(vh[19])
    classes.sort()
    X_val, y_val = collect_class_vs(vh[19])
    lb = LabelBinarizer()
    le = LabelEncoder()
    lb.fit(classes)
    le.fit(classes)
    y_val_enc = le.transform(y_val)
    Y_val = lb.transform(y_val)


    ##load test vectors / generate test vectors using CNN:
    cnn_name = 'epoch_19.model'
    oname = cnn_name.split('.')[0]+'_test_pred'
    ofile = os.path.join(C.obj_dir,oname)
    #T.save_model_predictions(os.path.join(C.model_dir,cnn_name), tdir=C.test_dir, 
    #                        ofile=ofile)
    test = T.load_obj(ofile)
    X_test, y_test = collect_class_vs(test)
    y_test_enc = le.transform(y_test)

    #Decision tree
    model_tree = tree.DecisionTreeClassifier()
    model_tree.fit(X_val, Y_val)
    Y_test = lb.transform(y_test)
    Y_test_pred = model_tree.predict(X_test)
    train_acc = accuracy_score(Y_val, model_tree.predict(X_val))
    test_acc = accuracy_score(Y_test, Y_test_pred)
    print('Train / Test accuracy decision tree      = {:.2%} / {:.2%}'.format(train_acc,test_acc))

    #SVM
    model_svm = SVC(kernel='linear')
    model_svm.fit(X_val, y_val_enc)
    y_test_pred = model_svm.predict(X_test)
    train_acc = accuracy_score(y_val_enc, model_svm.predict(X_val))
    test_acc = accuracy_score(y_test_enc, y_test_pred)
    print('Train / Test accuracylinear SVC          = {:.2%} / {:.2%}'.format(train_acc, test_acc))

    #SVM with RBF
    model_svm_rbf = SVC(kernel='rbf', C=1E-7, probability=True)
    model_svm_rbf.fit(X_val, y_val_enc)
    y_test_pred = model_svm_rbf.predict(X_test)
    train_acc = accuracy_score(y_val_enc, model_svm_rbf.predict(X_val))
    test_acc = accuracy_score(y_test_enc, y_test_pred)
    print('Train / Test accuracy RBF SVC            = {:.2%} / {:.2%}'.format(train_acc, test_acc))

    #SVM with RBF on SVD-transformed data
    wh, items = svd_project(vh[19], return_items=True) 
    Z_val, y_val = collect_class_vs(wh)
    y_val_enc = le.transform(y_val)
    model_svm_svd = SVC(kernel='rbf', C=1E-7, probability=True)
    model_svm_svd.fit(Z_val, y_val_enc)
    #convert test data, with the  normalisation and transformation from the training data
    X_mean = items['X_mean']
    X_std = items['X_std']
    test_n = {}
    for c in test: #for all classes:
        test_n[c] = (np.squeeze(test[c]) - X_mean) / X_std
    Z_test, y_test = collect_class_vs(test_n)
    Z_test = Z_test @ items['U']
    y_test_enc = le.transform(y_test)
    y_test_pred = model_svm_svd.predict(Z_test)
    train_acc = accuracy_score(y_val_enc, model_svm_svd.predict(Z_val))
    test_acc = accuracy_score(y_test_enc, y_test_pred)
    print('Train / Test accuracy  RBF SVC  with SVD = {:.2%} / {:.2%}'.format(train_acc, test_acc))

    #K-means clustering (unsupervised)
    classes = list(vs)
    classes.sort()
    n_clusters = len(classes)
    centersSKL = cluster.MiniBatchKMeans(n_clusters)
    centersSKL.fit(X_val) 
    y_pred = centersSKL.predict(X_val)
    #identify the clusters by class (hopefully)
    centroids = {} #for the predicted clusters
    radii = {}
    #find centroids, radii of detected clusters
    for k in set(y_pred):
        k_vecs = [X_val[i, :] for i in range(len(y_pred)) if y_pred[i]==k]
        centroids[k] = T.centroid(k_vecs)
        radii[k] = T.radius(centroids[k], k_vecs)
    #find centroids, radii of the true classes
    centroids_actual = {} 
    radii_actual = {}
    for k in set(y_val):
        c_vecs = [X_val[i, :] for i in range(len(y_val)) if y_val[i]==k]
        centroids_actual[k] = T.centroid(c_vecs)
        radii_actual[k] = T.radius(centroids_actual[k], c_vecs)
    cent_list = [centroids_actual[c] for c in classes]
    rad_list = [radii_actual[c] for c in classes]
    #match each detected cluster to closest class 
    #by minimising L2 norm of (centroid; radius) difference
    def cluster_metric(c1, c2, r1, r2):
        return T.dist(c1, c2) + T.dist(r1,r2)
    def find_nearest_cluster(centroid, radius, centroid_list, radius_list):
        (c1, r1) = (centroid, radius)
        return np.argmin([cluster_metric(c1,centroid_list[i],r1,radius_list[i]) for i in range(len(radius_list))])
    k_map = {}
    for k in centroids: #cluster number
        best_i = find_nearest_cluster(centroids[k], radii[k], cent_list, rad_list)
        k_map[k] = classes[best_i]
    c_map = {c:k for k,c in k_map.items()} 
    #training score:
    y_pred_cls = [k_map[k] for k in y_pred] #classes
    y_pred_enc = le.transform(y_pred_cls) #numbers
    train_acc = accuracy_score(y_val_enc, y_pred_enc)
    #test score:
    y_pred_cls = [k_map[k] for k in centersSKL.predict(X_test)]
    y_test_pred = le.transform(y_pred_cls) #numbers
    X_test, y_test = collect_class_vs(test) #get fresh y_test vectors
    y_test_enc = le.transform(y_test)
    test_acc = accuracy_score(y_test_enc, y_test_pred)
    print('Train / Test accuracy K-means = {:.2%} / {:.2%}'.format(train_acc, test_acc))

    from sklearn.metrics import confusion_matrix as conf_mat
    cm = conf_mat(y_test_enc, y_test_pred)
    fig = plot_confusion_matrix(cm,
                            classes,
                            title='Confusion matrix, ',
                            cmap=None,
                            normalize=True,
                            figsize=(12,8))
    fig.savefig('notebooks/images/best_confusion.png')
    main()