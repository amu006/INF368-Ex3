
def set_trainable_layers(model, n=None):
    """
    Unlocks the last n layers of a model for training. Freezes the remainder.
    If n is None, unlocks all layers
    """
    model.trainable = True
    N = len(model.layers)
    for i, l in enumerate(model.layers):
        l.trainable = i >= (N - n)
    return