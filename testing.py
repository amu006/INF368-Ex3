import config as C

from PIL import Image
import numpy as np
import pickle
import sys
from keras.models import load_model

from generators import paste

def save_obj(obj, fname ):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(fname ):
    with open(fname + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_model_predictions(model_name, 
                           tdir=C.val_dir, ofile='val_pred'):
    """ 
    Loads the specified model and predicts from data in tdir, 
    then pickles the predictions
    """
    base_model = load_model(model_name)
    vs = get_vectors(base_model, tdir)
    save_obj(vs, ofile)
    print('Predictions from {} --> {}'.format(model_name, ofile))
    return


def class_file(model, fname):
    img = np.array(Image.open(fname))/256
    return model.predict(np.expand_dims(paste(img), axis=0))

def dist(x,y):
    return np.linalg.norm(x-y)

# Calculate histogram of all distances from v in v1 to w in v2
def dist_hist(v1,v2):
    return [dist(v,w) for v in v1 for w in v2]

def centroid(vs):
    x0 = np.zeros_like(vs[0])
    for x in vs:
        x0 = np.add(x0,x)
    return x0/len(vs)

import os

# return the set of classes with vectors from model
def get_vectors(model, tdir=C.test_dir):
    classes = os.listdir(tdir)
    res = {}
    for c in classes:
           images = os.listdir(os.path.join(tdir,c))
           res[c] = [class_file(model, os.path.join(tdir,c,f)) for f in images]
    return res

# radius of a cluster, i.e. average or max distance from centroid
def radius(c, vs, avg=True):
    ds = [dist(c,v) for v in vs]
    if avg:
        return sum(ds)/len(ds)
    else:
        return max(ds)

def radius_nd(c, vs, avg=True):
    ds = [c - v for v in vs]
    if avg:
        return [np.mean(np.abs([d[i] for d in ds])) for i in range(len(c))]
    else:
        return [np.max(np.abs([d[i] for d in ds])) for i in range(len(c))]

# for backwards compatibility
def centroid_distances(vectors, outfile=sys.stdout):
    # load a bunch of images, calculate outputs
    res = {}
    for c in vectors:
           ds = dist_hist(vectors[c],vectors[c])
           res[c] = centroid(vectors[c])
           print(c[:25].ljust(26),' average radius: %.4f worst case diameter: %.4f' % (sum(ds)/len(ds), max(ds)), file=outfile)
    print('\nCentroid distances',file=outfile)
    for c in res:
        print(c[:25].ljust(26),end=': ', file=outfile)
        for n in res:
            print('  %.3f' % (dist(res[c],res[n])), end='', file=outfile)
        print(file=outfile)

# average radius and centroid distances for all test classes
def summarize(vectors, outfile=sys.stdout):
    cents = {}
    rads  = {}
    for c in vectors:
        cents[c] = centroid(vectors[c])
        rads[c] = radius(cents[c], vectors[c])
    for c in vectors:
        print(c[:25].ljust(26),' r=%.3f ' % rads[c], end='', file=outfile)
        for n in vectors:
            print('  %.3f' % dist(cents[c],cents[n]), end='', file=outfile)
        print(file=outfile)

# assign each input to nearest centroid and tally
def count_nearest_centroid(vectors):
    cents = {}
    for c in vectors:
        cents[c] = centroid(vectors[c])
    counts = {}
    for c in vectors:
        counts[c] = {}
        for x in vectors:
            counts[c][x] = 0
        for v in vectors[c]:
            # find closest centroid, and bump its count
            nearest = None
            mindist = 9999999
            for ct in cents:
                d = dist(v,cents[ct])
                if d < mindist:
                    nearest = ct
                    mindist = d
            counts[c][nearest] = counts[c][nearest] + 1 
    return counts

def accuracy_counts(cts, outfile=sys.stdout):
    correct = 0
    total   = 0
    for v in cts:
        correct = correct + cts[v][v]
        for w in cts:
            total = total + cts[v][w]
    print('Accuracy: %.3f' % (correct/total), file=outfile)
        
def confusion_counts(cts, outfile=sys.stdout):
    for v in cts:
        print(v[:25].ljust(26),end='', file=outfile)
        for w in cts:
            print(" %4d" % cts[v][w], end='', file=outfile)
        print(file=outfile)

# find the class and distance of the k nearest elements to tvec in refset
def find_nearest(refset, tvec, k=1):
    mindist = []
    for c in refset:
        for p in refset[c]:
            d = dist(p,tvec)
            if len(mindist)<k or d < max([x for x,v in mindist]):
                mindist.append((d,c))
                mindist.sort()
                mindist = mindist[:k]
    return mindist

# classify data in tdir using kNN with rdir as the reference
def knn_test(model, rdir, tdir, k=5):
    rvecs = get_vectors(model, rdir)
    tvecs = get_vectors(model, tdir)
    cmx = {}
    for x in tvecs:
        cmx[x] = {}
        for y in rvecs:
            cmx[x][y] = 0
    for c in tvecs:
        for v in tvecs[c]:
            xs = find_nearest(rvecs, v, k)
            rs = [v for c,v in xs]
            r = max(set(rs), key=rs.count)
            # if r != c: print(xs)
            cmx[c][r] = cmx[c][r]+1
    return cmx

# todo: PCA plots

# Default test to use
def run_test(model, tdir=C.test_dir, outfile=sys.stdout):
    vecs = get_vectors(model,tdir)

    print('Centroid dists:')
    centroid_distances(vecs)

    print('Summarize:')
    summarize(vecs)

    c = count_nearest_centroid(vecs)
    accuracy_counts(c, outfile=outfile)

    print('Confusion matrix:')
    confusion_counts(c)
