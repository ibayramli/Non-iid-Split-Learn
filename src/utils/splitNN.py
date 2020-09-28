from itertools import chain
from .batchnorm import SplitBatchNorm2d
import torch


class SplitNN(torch.nn.Module):
    def __init__(self, clients, server):
        """
        clients: a dict/list of client_id: client_model pairs
        
        client_optims: a dict/list of client_id: client_optim pairs
        
        server: the server model
        
        server_optim: the server optimizier
        """
        super(SplitNN, self).__init__()
        self.clients = clients
        self.server = server 
        self.out_clients = {}
        self.in_server = {}
        
    def forward(self, x, client_id):
        """
        x: input tensor
        
        client_id: client_id of the client to receive the input
        """
        client = self.clients[client_id]
        server = self.server
        
        # for proper tracking / use of running stats
        self._set_batchnorm_clt(client_id)
        
        # out_client is a pointer to the activations physically located in client
        out_client = client.forward(x)
       
        # since out_client is a pointer, we are sending the server a pointer. The 
        # activations tensor itself is still located in the client. To relocate it, 
        # we call remote_get() in the server. 
        
        x_server = out_client.copy().send(server.location).remote_get()
        out_server = server.forward(x_server)
        
        # These will be needed for backprop
        self.out_clients[client_id] = out_client
        self.in_server[client_id] = x_server
        
        return out_server

    def backward_clt(self, client_id):
        client = self.clients[client_id]
        out_client = self.out_clients[client_id]
        in_server = self.in_server[client_id]
        
        grad_server = in_server.grad.copy().move(client.location)
        out_client.backward(grad_server)
        
    def _set_batchnorm_clt(self, client_id):
        client = self.clients[client_id]
        server = self.server

        for module in chain(client.modules(), server.modules()):
            if isinstance(module, SplitBatchNorm2d):
                if module.track_running_stats:
                    module.client_id = client_id


class SplitOptimizer:
    def __init__(self, client_optims=None, server_optim=None):
        self.clients = client_optims
        self.server = server_optim
        
    def step(self, client_id):    
        client = self.clients[client_id]
        server = self.server
        
        if client is not None:
            client.step()
        if server is not None:
            server.step()
        
    def zero_grad(self, client_id):
        if self.clients[client_id] is not None:
            self.clients[client_id].zero_grad()
        if self.server is not None:
            self.server.zero_grad()


class SplitScheduler:
    def __init__(self, client_schedulers=None, server_scheduler=None):
        self.clients = client_schedulers
        self.server = server_scheduler
        
    def step(self):
        if self.clients is not None:
            for c in self.clients.keys():
                self.clients[c].step()
        if self.server is not None:
            self.server.step()
            
