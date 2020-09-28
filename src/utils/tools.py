from .batchnorm import SplitBatchNorm2d

import syft as sy
import torch
import itertools
import sys
import gc

# TODO: Fix Visdom

#import visdom
#
#
#loss_plot = vis.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1,)).cpu(),
#            opts={
#                "xlabel":"Iteration",
#                "ylabel":"Loss",
#                "title":"Loss over time",
#                "legend":["Classification loss"],
#                "layoutopts":{
#                    "plotly": {
#                        "yaxis": { "type": "log" }
#                        }
#                    }
#                })
#
#acc_plot = vis.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1,)).cpu(),
#            opts={
#                "xlabel":"Epoch",
#                "ylabel":"Accuracy",
#                "title":"Accuracy over epochs",
#                "legend":["Classification accuracy"]
#                })


#class VisdomLogger:
#    def __init__(self, visdom_server, visdom_port):
#        self.server = visdom_server
#        self.port = visdom_port
#        self.vis = visdom.Visdom(server=self.server, port=self.port)
#      
#    def plot(self, stat, num_iter, plot_fn):
#        self.vis.line(
#                     X=torch.tensor([num_iter]).cpu(),
#                     Y=stat, # must be a tensor
#                     win=plot_fn,
#                     update='append')
#

def init_worker(worker_id, model, hook, device, finetune=None, requires_grad=True):
    location = sy.VirtualWorker(hook, id=worker_id)
    model = model.copy()
    
    if finetune is not None:
        state = torch.load(finetune)
        if state['batchnorm_stats']:
            batchnorm_stats = state['batchnorm_stats']
            for i, m in enumerate(server.modules()):
                if isinstance(m, SplitBatchNorm2d):
                    m.running_means = batchnorm_stats['running_means'][i]
                    m.running_vars = batchnorm_stats['running_vars'][i]
        
        model_state = state['model_state']
        model.load_state_dict(model_state)
        
    for param in model.parameters():
        param.requires_grad = requires_grad

    model.to(device)
    model.send(location)
    
    return model

def dict2string(d, key='%s', value='%.5f'):
    fmt = key + '=' + value
    return ', '.join(fmt % i for i in d.items())

def save(path, iteration, model, optimizer, **kwargs):
    model_state = None
    optimizer_state = None
    if model is not None:
        model_state = model.state_dict()
    if optimizer is not None:
        optimizer_state = optimizer.state_dict()
    
    batchnorm_stats = {
        'running_means': [],
        'running_vars': [],
    }
    
    for m in model.modules():
        if isinstance(m, SplitBatchNorm2d):
            running_means = {}
            for k in m.running_means.keys():
                running_means[k] = m.running_means[k].copy().get()
            
            running_vars = {}
            for k in m.running_vars.keys():
                running_vars[k] = m.running_vars[k].copy().get()
                
            batchnorm_stats['running_means'].append(running_means)
            batchnorm_stats['running_vars'].append(running_vars)
    
    torch.save(
        dict(iteration=iteration,
             model_state=model_state,
             optimizer_state=optimizer_state,
             # Only save batchnorm_stats if model has SplitBatchNorm2d instances
             batchnorm_stats=(batchnorm_stats if batchnorm_stats['running_means'] else None),
             **kwargs),
        path
    )
    
# Source: https://szymonmaszke.github.io/torchfunc/_modules/torchfunc.html#sizeof
def sizeof(obj):
    '''
    Returns the size of the torch object in bytes. 
    '''
    if torch.is_tensor(obj):
        return obj.element_size() * obj.numel()

    elif isinstance(obj, torch.nn.Module):
        return sum(
            sizeof(tensor)
            for tensor in itertools.chain(obj.buffers(), obj.parameters())
        )
    else:
        return sys.getsizeof(obj)

def cuda_mem_stats(device_id):
    t = torch.cuda.get_device_properties(device_id).total_memory / 1e+9
    c = torch.cuda.memory_cached(device_id) / 1e+9
    a = torch.cuda.memory_allocated(device_id) / 1e+9
    
    print('Total Memory (GB):', t)
    print('Cache (GB):', c)
    print('Allocated (GB):', a)
    
    return t, c, a

def mem_report():
    objects = []
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            objects.append((type(obj), obj.size()))
    
    return objects

