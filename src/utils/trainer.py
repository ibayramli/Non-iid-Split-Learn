from .batchnorm import SplitBatchNorm2d
from .splitNN import *
from .tools import *
from torch import nn, optim
from copy import deepcopy

import syft as sy
import torch
import joblib
import os


LOG_NTH_SAMPLE = 10

class SplitTrainer:
    def __init__(self, client_ids, splitNN, split_optim, split_scheduler, dataloaders, 
                 criterion, num_epochs, save_path, device, 
                 visdom_logger=None, num_classes=100, verbose=1):
        """
        client_ids: a list of client ids,
        
        splitNN: an instance of SplitNetwork
        
        dataloader: a dict of dataloaders of the following form:
                    {
                        client_id: {
                            'train': train_dataloader
                            'val': val_dataloader
                        }
                    }

        criterion: loss criterion
        
        num_epochs: number of training and validation epochs 
        
        TODO: add support for more metrics
        
        TODO: implement Visdom Logger
        
        TODO: Find out why the number of objects in remote clients is increasing
        
        """
        self.client_ids = client_ids
        self.network = splitNN
        self.optim = split_optim
        self.scheduler = split_scheduler
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.save_path = save_path
        self.device = device
        self.num_classes = num_classes
        self.vis_logger = visdom_logger # TODO: Implement Visdom Logger
        self.verbose = verbose
        self.stats = self._init_stats()
        self.cum_epochs = 0

        
    def train_sequential(self, train_clts=None, val_clts=None):
        """
        train_clts: clients to train on,
        
        val_clts: clients to validate on
        """
        train_clts = self._validate_clts(train_clts)
        val_clts = self._validate_clts(val_clts)
            
        for client_id in train_clts:
            for epoch in range(self.num_epochs):
                self.cum_epochs += 1
                print('Training client {}, epoch {}, lr: {}\n'.format(client_id, epoch + 1, self.optim.server.param_groups[0]['lr'],))

                train_metrics = self._run_epoch('train', client_id)
                self._log('train', 'loss', client_id, epoch + 1, train_metrics['loss'])
                if self.verbose > 0:
                    print('Train results for client {}, epoch {}: {}\n'.format(client_id, epoch + 1, dict2string(train_metrics)))
                    print('Validating client {}\'s model for epoch {}\n'.format(client_id, epoch + 1))
                    
                self.validate(val_clts)
                
                self.scheduler.step()
                
                self._save('client', self.network.clients[client_id], 
                           self.optim.clients[client_id], self.cum_epochs, client_id)
                self._save('server', self.network.server, 
                           self.optim.server, self.cum_epochs, client_id)
        
        joblib.dump(self.stats, os.path.join(self.save_path, 'stats.pkl'))
        return self.stats

    def train_alternate(self, train_clts, val_clts):
        iter_train = {}
        server_location = self.network.server.location
        
        client_weights = {}
        train_size = sum([self.dataloaders[clt]['train'].batch_size for clt in train_clts])
        for clt in train_clts:
            client_weights[clt] = self.dataloaders[clt]['train'].batch_size / train_size
        
        n_params_server = len(list(self.network.server.parameters()))
        
        for epoch in range(self.num_epochs):
            print(
                'Training epoch {}. Learning rate: {} {}'.format(epoch + 1, self.optim.server.param_groups[0]['lr'], '-'*50)
            )
            self.cum_epochs += 1
            
            self.network.server.train()
            for c in train_clts:
                iter_train[c] = iter(self.dataloaders[c]['train'])
                self.network.clients[c].train()
                
            running_metrics = { 'accuracy': {}, 'loss': {} }
            for c in train_clts + ['global']:
                for metric in running_metrics.keys():
                    running_metrics[metric][c] = 0
            
            counter = 0
            while iter_train:
                counter += 1
                accum_grads = [0 for i in range(n_params_server)]
                for client_id in list(iter_train):
                    client_iter = iter_train[client_id]
                    try:
                        data = next(client_iter)
                    except StopIteration:
                        iter_train.pop(client_id)
                        continue 

                    image, label = data
                    client_location = self.network.clients[client_id].location
                    image = image.to(self.device).send(client_location)
                    label = label.to(self.device).send(server_location)

                    self.optim.server.zero_grad()
                    self.optim.clients[client_id].zero_grad()
                    
                    pred = self.network.forward(image, client_id)
                    loss = self.criterion(pred, label) * client_weights[client_id]
                    
                    # sets up gradients of a single forward pass for both server and clients
                    loss.backward()
                    self.network.backward_clt(client_id)

                    # Accumulate gradients from all the clients
                    for i, p in enumerate(self.network.server.parameters()):
                        grad = p.grad.copy().get()
                        accum_grads[i] += grad

                    pred, loss = pred.detach(), loss.detach().get().item()
                    # classification case
                    if self.num_classes > 1:
                        pred_class = torch.argmax(pred, axis=1)
                        acc = torch.eq(pred_class, label).float().mean().get().item() * 100
                        
                    # Record training metrics
                    n = len(self.dataloaders[client_id]['train'])
                    running_metrics['accuracy'][client_id] += acc / n
                    running_metrics['loss'][client_id] += loss / n
                    
                    running_metrics['accuracy']['global'] += client_weights[client_id] * acc / n
                    running_metrics['loss']['global'] += client_weights[client_id] * loss / n
                    
                    # Print progress 
                    if self.verbose > 1:
                        if counter % LOG_NTH_SAMPLE == 0:
                            print(
                                 'Client {}. Loss: {:.3f}, accuracy: {:.2f}%'.format(
                                     client_id, loss, acc
                                 )
                                  )
                            if client_id == list(iter_train)[-1]:
                                print('\n')
                
                if not iter_train:
                    continue

                # Add gradients of the server
                for i, p in enumerate(self.network.server.parameters()):
                    new_grad = accum_grads[i].send(p.location)
                    # We use += -p.grad + new_grad instead of direct assignment to bypass a bug with 
                    # PySyft where new_grad becomes a tensors of zeros when read by .grad.setter.
                    p.grad += -p.grad + new_grad
                    
                # Add gradients of the clients
                for params in zip(
                    *list(
                        clt.parameters() for clt in self.network.clients.values()
                    )
                ):
                    sum_grads = 0.
                    for client_id, p in enumerate(params):
                        p_grad = p.grad.copy().get()
                        sum_grads += p_grad
                    
                    for p in params:
                        new_grad = sum_grads.send(p.location)
                        p.grad += -p.grad + new_grad
                
                # Optimize server with summed grads
                self.optim.server.step()
                
                # Optimize clients with summed grads
                for c in train_clts:
                    self.optim.clients[c].step()
              
            
                # Save training stats
                for c in train_clts + ['global']:
                    for metric in running_metrics.keys():
                        self._log('train', metric, c, self.cum_epochs, running_metrics[metric][c])
              
            # Run validation
            self.validate(val_clts)
            
            # Call the scheduler
            self.scheduler.step()
            
            for c in train_clts:
                self._save('client', self.network.clients[c], 
                           self.optim.clients[c], self.cum_epochs, c)

            self._save('server', self.network.server, 
                       self.optim.server, self.cum_epochs, client_id)
            
        return self.stats
    
    def validate(self, val_clts):
        client_weights = {}
        val_size = sum([self.dataloaders[clt]['val'].batch_size for clt in val_clts])
        for clt in val_clts:
            client_weights[clt] = self.dataloaders[clt]['val'].batch_size / val_size
        
        global_metrics = {}
        for c in val_clts:
            val_metrics = self._run_epoch('val', c)
            if self.verbose > 0:
                print('Val results for client {}, epoch {}: {}\n'.format(c, self.cum_epochs, dict2string(val_metrics)))
            
            for metric in val_metrics.keys():
                if metric in global_metrics.keys():
                    global_metrics[metric] += client_weights[c] * val_metrics[metric]
                else:
                    global_metrics[metric] = client_weights[c] * val_metrics[metric]

                self._log('val', metric, c, self.cum_epochs, val_metrics[metric])
                
                if c == val_clts[-1]: # on last iteration record global metrics
                    self._log('val', metric, 'global', self.cum_epochs, global_metrics[metric])
            
        print('Val results global, epoch {}: {}\n'.format(self.cum_epochs, dict2string(global_metrics)))
    
    def _run_epoch(self, mode, client_id):
        assert mode in ['train', 'val']
        if mode == 'train':
            self.network.clients[client_id].train()
            self.network.server.train()
        else:
            self.network.clients[client_id].eval()
            self.network.server.eval()

        client_location = self.network.clients[client_id].location
        server_location = self.network.server.location
        dataloader = self.dataloaders[client_id][mode]
        num_samples = len(dataloader)
        
        running_acc = 0
        running_loss = 0
        metrics = {}
        for i, data in enumerate(dataloader):
            image, label = data
            image = image.to(self.device).send(client_location)
            label = label.to(self.device).send(server_location)
            
            if mode == 'train':
                pred, loss = self._train_iter(image, label, client_id)
            else:
                pred, loss = self._validate_iter(image, label, client_id)

            running_loss += loss
            # classification case
            if self.num_classes > 1:
                pred_class = torch.argmax(pred, axis=1)
                acc = torch.eq(pred_class, label).float().mean().get().item() * 100
                running_acc += acc
            if self.verbose > 1:
                if i % LOG_NTH_SAMPLE == 0:
                    print(
                        'Sample {} out of {}. Loss: {:.3f}, accuracy: {:.2f}%'.format(
                            i, num_samples, loss, acc
                        )
                         ) 
    
        metrics['loss'] = running_loss / num_samples
        if self.num_classes > 1:
            metrics['accuracy'] = running_acc / num_samples
        return metrics
    
    def _train_iter(self, x, label, client_id, step=True):
        self.optim.zero_grad(client_id)
        
        pred = self.network.forward(x, client_id)
        loss = self.criterion(pred, label)
        
        # backprops both server and client weights 
        loss.backward()
        self.network.backward_clt(client_id)
        
        self.optim.step(client_id)
        
        return pred.detach(), loss.detach().get().item()
    
    def _validate_iter(self, x, label, client_id):
        with torch.no_grad():
            pred = self.network.forward(x, client_id)
            loss = self.criterion(pred, label)
        
        return pred.detach(), loss.detach().get().item()
    
    def _validate_clts(self, clts):
        if clts is None:
            return self.client_ids
        for c in clts:
            if c not in self.client_ids:
                raise ValueError('Cannot train or validate on a client not part of the network')
        else:
            return clts
    
    def _init_stats(self):
        stats = { 
            'train': { 'loss': {} }, 
            'val': { 'loss': {} } 
        }
        
        if self.num_classes > 1:
            stats['val']['accuracy'] = {}
            stats['train']['accuracy'] = {}
        
        for mode in stats.keys():
            for stat in stats[mode].keys():
                for client in self.client_ids + ['global']:
                    stats[mode][stat][client] = {}
        
        return stats
        
    def _log(self, mode, stat, client_id, epoch, stat_val):
        self.stats[mode][stat][client_id][epoch] = stat_val

    def _save(self, mode, model, optim, epoch, client_id):
        if self.save_path is None:
            return
        assert mode in ['client', 'server']
        assert client_id in self.client_ids    
        
        # Inefficiency Note: 
        # Due to a (now-reported) bug in PySyft source code, self.network.clients[client_id].copy().get()
        # gives an error. Here, a workaround is used to prevent from client models being erased from the 
        # remote worker during save. The location of the client model is saved before .get() (which deletes the 
        # client model from the original location) and the localized model is sent back to that worker after save
        # Once the bug with .copy().get() is resolved, this code should be modified for efficiency reasons
        # See the issue at https://github.com/OpenMined/PySyft/issues/3845
        
        location = model.location
        model = model.get()
        filename = '_'.join([mode, 'trained_on', 'client', str(client_id)]) + '.pth'
        path = os.path.join(self.save_path, 'epoch_' + str(epoch))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, filename)
        stats = self.stats
        
        save(
            path,
            epoch, 
            model,
            optim,
            stats=stats
        )
        
        if mode == 'client':
            self.network.clients[client_id] = model.send(location)
        else:
            self.network.server = model.send(location)
                          
def split_train(client_ids, data_paths, server_model, client_model, 
                hook, optim, optim_params, criterion, dataloader_fn,
                dataloader_params, save_path, num_epochs, verbose, cuda=True):
    
    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
    
    # Workers are automatically put on to the appropriate devices and clients
    server = init_worker('server', server_model, hook, device)
    clients = {}
    for c in client_ids:
        clients[c] = init_worker('client_'+str(c), client_model, hook, device)
    
    splitNN = SplitNN(clients, server)
    criterion = criterion.to(device)
    
    server_optim = optim(splitNN.server.parameters(), **optim_params)
    client_optims = {}
    for c, client in splitNN.clients.items():
        client_optims[c] = optim(client.parameters(), **optim_params)
    
    split_optim = SplitOptimizer(client_optims, server_optim)
    
    # Get the data
    dataloaders = {}
    for c in client_ids:
        dataloaders[c] = {}
        dataloaders[c]['train'] = dataloader_fn(data_paths[c]['train'], **dataloader_params)
        dataloaders[c]['val'] = dataloader_fn(data_paths[c]['val'], **dataloader_params)
        
    trainer = SplitTrainer(client_ids=client_ids, splitNN=splitNN, split_optim=split_optim,
                           dataloaders=dataloaders, criterion=criterion, num_epochs=num_epochs, 
                           save_path=save_path, verbose=verbose, device=device)
    stats = trainer.train()
    
    return stats

