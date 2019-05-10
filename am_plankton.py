
def set_trainable_layers(model, n=None, mute=False):
    """
    Unlocks the last n layers of a model for training. Freezes the remainder.
    If n is None, unlocks all layers
    """
    if not mute:
        log_str = 'Setting top {} layers of base model trainable'.format(n)
        print(log_str)
    model.trainable = True
    N = len(model.layers)
    for i, l in enumerate(model.layers):
        l.trainable = i >= (N - n)
    if not mute:
        return log_str
    else:
        return
    
def set_trainable_layers_smart(model, p=1000000, mute=False):
    """
    Unlocks layers of a model from the top, until there are >= p more trainable parameters than before
    
    Inputs:
        model - a model (usually the base_model)
        p - integer, number of extra parameters to unlock
    """
    new_params = 0
    unlock_count = 0
    for i in range(len(model.layers) - 1, -1, -1):
        l = model.layers[i]
        if l.trainable is False:
            if new_params < p:
                l.trainable = True
                unlock_count += 1
                new_params += l.count_params() 
    if not mute:
        print('{} layers unlocked to give an additional {} parameters'.format(unlock_count, new_params))
    return
        