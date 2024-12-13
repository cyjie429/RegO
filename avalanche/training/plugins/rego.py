import copy
from collections import defaultdict
import torch
import math
import random
from avalanche.models import avalanche_forward, MultiTaskModule
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from torch.utils.data import DataLoader
import collections
import numpy as np
from torch.nn.functional import cosine_similarity
from scipy.optimize import fsolve
from torch.nn import CrossEntropyLoss


class RegOPlugin(StrategyPlugin):

    def __init__(self, alpha=1, temperature=2, model=None, ef_thresh=None, importnat_thresh=None):
        """
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        """

        super().__init__()
        self.target2ti = {"0":-1, "1":1, "2":-1, "3":-1, "4":-1, "5":-1, "6":1, "7":1, "8":-1, "9":1, "10":-1}
        dtype = torch.cuda.FloatTensor  # run on GPU
        self.real_importances = defaultdict(list)
        self.real_filters = defaultdict(list)
        self.fake_importances = defaultdict(list)
        self.fake_filters = defaultdict(list)
        self.pre_representation = None
        self.old_gradient = {}
        self.ef_thresh = ef_thresh 
        self.importnat_thresh = importnat_thresh
        self.criterion = CrossEntropyLoss()
        """ In Avalanche, targets of different experiences are not ordered. 
        As a result, some units may be allocated even though their 
        corresponding class has never been seen by the model.
        Knowledge distillation uses only units corresponding to old classes. 
        """

    def compute_importances(self, model, criterion, optimizer,
                            dataset, device, batch_size):
        """
        Compute Network importance for each neuron
        """

        model.eval()
        real_importances = zerolike_params_dict(model)
        fake_importances = zerolike_params_dict(model)
        real_num = 0
        fake_num = 0
        dataloader = DataLoader(dataset, batch_size=batch_size)
            
        for i, batch in enumerate(dataloader):
            x, y, task_labels = batch[0], batch[1], batch[-1]
            x, y = x.to(device), y.to(device)
            mapping_tensor = torch.tensor([self.target2ti[str(i)] for i in range(11)])
            mapped_y = mapping_tensor[y]
            
            real_idx = torch.nonzero(mapped_y == -1).squeeze()
            fake_idx = torch.nonzero(mapped_y == 1).squeeze()
            real_x = x[real_idx]
            fake_x = x[fake_idx]
            real_y = y[real_idx]
            fake_y = y[fake_idx]
            real_num += len(real_idx)
            fake_num += len(fake_idx)
            
            optimizer.zero_grad()
            real_out = avalanche_forward(model, real_x, task_labels)
            real_loss = criterion(real_out, real_y)
            real_loss.backward()

            for (k1, p), (k2, imp) in zip(model.named_parameters(), real_importances):
                assert (k1 == k2)
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)
                    
            optimizer.zero_grad()
            fake_out = avalanche_forward(model, fake_x, task_labels)
            fake_loss = criterion(fake_out, fake_y)
            fake_loss.backward()

            for (k1, p), (k2, imp) in zip(model.named_parameters(), fake_importances):
                assert (k1 == k2)
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)
                    
        # average over mini batch length
        for _, imp in real_importances:
            imp /= real_num
            
        for _, imp in fake_importances:
            imp /= fake_num

        return real_importances, fake_importances
    
    # TODO: 根据总的数值等于1来设计S，并且获得艾宾浩斯遗忘权重
    def set_ebbinghaus_forgetting_weight(self, task_num):
        
        def equation(x, task_num):
            formula = 0
            for i in range(1, task_num+1):
                formula += np.exp(-i/x)
            return formula - 1
        
        initial_guess = 1.0  # 初始猜测值
        solution = fsolve(equation, initial_guess, args=(task_num))
        return list(reversed([np.exp(-i/solution[0]) for i in range(1, task_num+1)]))
        
    # 这个是在optimizer之前
    def before_update(self, strategy, **kwargs):
        fake_batch_idx = []
        real_batch_idx = []
        
        for idx, s_output in enumerate(strategy.mb_output):
            if self.target2ti[str(int(strategy.mb_y[idx]))] == -1:
                fake_batch_idx.append(idx)
            else:
                real_batch_idx.append(idx)
        real_num = len(real_batch_idx)
        fake_num = len(fake_batch_idx)
        
        
        with torch.no_grad():
            if strategy.experience.current_experience != 0:
                for n, w in strategy.model.named_parameters(): 

                    if n == "module.weight":
                        original_shape = w.grad.shape
                        current_grad = w.grad.clone()
                        
                        ebbinghaus_forgetting_weight = self.set_ebbinghaus_forgetting_weight(strategy.experience.current_experience)
                        # print(ebbinghaus_forgetting_weight)
                        ebbinghaus_filter = torch.zeros_like(self.old_gradient[n])
                        
                        for _ in range(strategy.experience.current_experience):
                            ebbinghaus_filter += (self.real_filters[_][n] | (self.fake_filters[_][n]/2).to(torch.int)) * ebbinghaus_forgetting_weight[_]

                        ebbinghaus_grad_filter = torch.where(ebbinghaus_filter > self.ef_thresh, 1, 0).to(torch.int)

                        pre_real_grad_filter = torch.zeros_like(self.old_gradient[n]).to(torch.int)
                        for _ in range(strategy.experience.current_experience):
                            pre_real_grad_filter = pre_real_grad_filter | self.real_filters[_][n]
                        pre_fake_grad_filter = torch.zeros_like(self.old_gradient[n]).to(torch.int)
                        for _ in range(strategy.experience.current_experience):
                            pre_fake_grad_filter = pre_fake_grad_filter | self.fake_filters[_][n]
                        grad_filter = pre_real_grad_filter + pre_fake_grad_filter
                        # Forgetting Matrix
                        grad_filter = torch.mul(grad_filter, ebbinghaus_grad_filter) 
                        # Region A
                        zero_zone = torch.zeros_like(grad_filter)
                        zero_zone[grad_filter == 0] = 1
                        # Region B
                        one_zone = torch.zeros_like(grad_filter)
                        one_zone[grad_filter == 1] = 1
                        # Region C
                        two_zone = torch.zeros_like(grad_filter)
                        two_zone[grad_filter == 2] = 1
                        # Region D
                        three_zone = torch.zeros_like(grad_filter)
                        three_zone[grad_filter == 3] = 1
                        w.grad.zero_()
                        # print(-11, torch.sum(three_zone == 1).item())
                        w.grad = torch.mul(current_grad, zero_zone)
                        # the optimization method for region B
                        one_old_grad = self.old_gradient[n].view(-1).clone()
                        one_g_c_g_o = torch.dot(current_grad.view(-1), one_old_grad)
                        one_proj_grad = one_g_c_g_o * one_old_grad / (one_old_grad.norm() ** 2)
                        w.grad += torch.mul(one_proj_grad.view(original_shape), one_zone) 
                        # the optimization method for region C
                        two_old_grad = self.old_gradient[n].view(-1).clone()
                        two_g_c_g_o = torch.dot(current_grad.view(-1), two_old_grad)
                        two_proj_grad = two_g_c_g_o * two_old_grad / (two_old_grad.norm() ** 2)
                        two_orthogonal_grad = current_grad - two_proj_grad.view(original_shape) 
                        w.grad += torch.mul(two_orthogonal_grad, two_zone) 
                        # the optimization method for region D
                        r_f_g = (real_num / strategy.train_mb_size) * one_proj_grad.view(original_shape) + (fake_num / strategy.train_mb_size) * two_orthogonal_grad
                        w.grad += torch.mul(r_f_g, three_zone)
            
            # save current gradient as old gradient
            for n, w in strategy.model.named_parameters():
                self.old_gradient[n] = w.grad.clone()


    def after_training_exp(self, strategy, **kwargs):
        """
            update importances
        """ 
        # compute importances
        real_importances, fake_importances = self.compute_importances(strategy.model,
                                               strategy._criterion,
                                               strategy.optimizer,
                                               strategy.experience.dataset,
                                               strategy.device,
                                               strategy.train_mb_size)
        real_filters = {}
        for i in range(len(real_importances)):
            if "weight" in real_importances[i][0]:
                threshold = real_importances[i][1].quantile(self.importnat_thresh)
                filter_matrix = torch.where(real_importances[i][1] > threshold, 1, 0)
                real_filters[real_importances[i][0]] = filter_matrix     
        self.real_importances[strategy.experience.current_experience] = real_importances
        self.real_filters[strategy.experience.current_experience] = real_filters
        
        fake_filters = {}
        for i in range(len(fake_importances)):
            if "weight" in fake_importances[i][0]:
                threshold = fake_importances[i][1].quantile(self.importnat_thresh)
                filter_matrix = torch.where(fake_importances[i][1] > threshold, 2, 0)
                fake_filters[fake_importances[i][0]] = filter_matrix     
        self.fake_importances[strategy.experience.current_experience] = fake_importances
        self.fake_filters[strategy.experience.current_experience] = fake_filters