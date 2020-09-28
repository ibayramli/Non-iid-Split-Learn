# Built upon: https://github.com/ptrblck/pytorch_misc/edit/master/batch_norm_manual.py
import torch
import torch.nn as nn


class SplitBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(SplitBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.client_id = None
        self.running_means = {}
        self.running_vars = {}

    def forward(self, input):
        self._check_input_dim(input)
        
        client_id = self.client_id
        self._validate_clt(client_id)
        
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training or not self.track_running_stats:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.shape[1]
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_means[client_id] = exponential_average_factor * mean \
                        + (1 - exponential_average_factor) * self.running_means[client_id]
                    # update running_var with unbiased var
                    self.running_vars[client_id] = exponential_average_factor * var * n / (n - 1)\
                        + (1 - exponential_average_factor) * self.running_vars[client_id]
        else:
            mean = self.running_means[client_id]
            var = self.running_vars[client_id]

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input

    def _validate_clt(self, client_id):
        if client_id is None:
            return
        if client_id not in self.running_means.keys():
            self.running_means[client_id] = torch.zeros(self.num_features).send(self.weight.location)
        if client_id not in self.running_vars.keys():
            self.running_vars[client_id] = torch.ones(self.num_features).send(self.weight.location)
    
