import torch
import pyswarms as ps
import numpy as np
import math

from explainer.BPSO_ES import BPSO_ES
from explainer.GA_parallel import GA_parallel


class Block():
    def __init__(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point
        self.size = max(self.end_point[0] - self.start_point[0], self.end_point[1] - self.start_point[1])
    

class SegmentIdentificationSearch():
    def __init__(self, net, device):
        self.net = net
        self.device = device
        self.net.eval()
        self.max_batch_size = None

        # define a smoother kernel
        smoother_kernel_1d = torch.nn.Conv1d(1,1,3,stride = 1, bias = False)
        a = torch.ones(1, 1,3)*(-1)
        a[0, 0, 1] = 2
        for _ in smoother_kernel_1d.parameters():
            _.data = a
            _.requires_grad = False
        self.smoother_kernel_1d = smoother_kernel_1d
        self.smoother_kernel_1d.to(self.device)

    
    def block_max_size(self):
        sizes = []
        for i in range(len(self.blocks)):
            sizes.append(self.blocks[i].size)
        return np.max(sizes)

    def next_meshing(self, blocks, selected_block):
        new_blocks= {}
        block_start_index = 0
        for erased_block in selected_block:
            sub_block = self.meshing(blocks[erased_block], scaler = [2, 2], block_start_index = block_start_index)
            block_start_index += len(sub_block)
            new_blocks.update(sub_block)
        return new_blocks
    
    def meshing(self, block, scaler = [2, 2], block_start_index = 0):

        len_dim1 = block.end_point[0] - block.start_point[0]
        len_dim2 = block.end_point[1] - block.start_point[1]

        sub_len_dim1 = math.ceil(len_dim1 / scaler[0])
        sub_len_dim2 = math.ceil(len_dim2 / scaler[1])
        split_point_dim1 = np.array([i for i in range(0, len_dim1, sub_len_dim1)] + [len_dim1])
        split_point_dim2 = np.array([i for i in range(0, len_dim2, sub_len_dim2)] + [len_dim2])
        split_point_dim1, split_point_dim2 = np.meshgrid(split_point_dim1, split_point_dim2)

        subblock = {}
        block_start_index = block_start_index
        for grid_dim1 in range(1, split_point_dim1.shape[0]):
            for grid_dim2 in range(1, split_point_dim2.shape[1]):
                
                start_point = (split_point_dim1[grid_dim1-1, grid_dim2-1] + block.start_point[0], split_point_dim2[grid_dim1-1, grid_dim2-1] + block.start_point[1])
                end_point = (split_point_dim1[grid_dim1, grid_dim2] + block.start_point[0], split_point_dim2[grid_dim1, grid_dim2] + block.start_point[1])

                subblock[block_start_index] = Block(start_point = start_point, end_point = end_point)
                block_start_index += 1
        return subblock

    def block_to_mask(self, best_selected_blocks):
        best_mask = np.ones(self.input_dim)
        for block_ind in best_selected_blocks:
            start_point, end_point = self.blocks[block_ind].start_point, self.blocks[block_ind].end_point
            best_mask[start_point[0]:end_point[0], start_point[1]:end_point[1]] = 0
        return best_mask

    def binary_to_mask(self, x):
        masks = np.ones((x.shape[0], *self.input_dim))
        idx = 0
        for particle in x:
            selected_block = np.nonzero(particle)[0]
            for block_ind in selected_block:
                start_point, end_point = self.blocks[block_ind].start_point, self.blocks[block_ind].end_point
                masks[idx, start_point[0]:end_point[0], start_point[1]:end_point[1]] = 0
            idx +=1
        return masks

    @torch.no_grad()
    def smoother_measure(self, mask):
        mask = torch.from_numpy(mask).to(torch.float).to(self.device)
        mask_for_smooth_pently = torch.transpose(mask, 1, 2).unsqueeze(2).reshape(-1, 1, mask.shape[1])
        smooth_pently = self.smoother_kernel_1d(mask_for_smooth_pently.to(torch.float)).cpu().abs().sum(dim = (1, 2)).detach().numpy()
        smooth_pently = smooth_pently.reshape(mask.shape[0], -1).sum(axis = 1)/1000
        return smooth_pently

    @torch.no_grad()
    def pred_score_probability(self, input, mask):
        x_o = input * mask
        perturbed_samples = self.perturb_func(x_o, mask)
        samples_prediction = self.net.forward(perturbed_samples)
        for _ in range(9):
            perturbed_samples = self.perturb_func(x_o, mask, batch_size = 1)
            samples_prediction += self.net.forward(perturbed_samples)
        return samples_prediction.squeeze()[self.origin_pred_label].item()/10, torch.argmax(samples_prediction, dim = 1).item()
    
    @torch.no_grad()
    def sparse_measure(self, mask):
        return mask.shape[1] * mask.shape[2] - mask.sum(dim = (1, 2)).cpu().detach().numpy()

    def get_max_batch_size(self, x_o, mask):
        for i in range(1, 101):
            try:
                self.genmodel.sample(x_o[:i, :, :], mask[:i, :, :])
                self.max_batch_size = i
            except:
                break
        
        self.max_batch_size = int(self.max_batch_size * 0.8) if self.max_batch_size < 100 else self.max_batch_size
    
    @torch.no_grad()
    def fitnessfun(self, x, input, **kwargs):
        # x are binary variable for features. 
        mask = self.binary_to_mask(x)

        mask = torch.from_numpy(mask).to(torch.float).to(self.device)
        mask_for_smooth_pently = torch.transpose(mask, 1, 2).unsqueeze(2).reshape(-1, 1, mask.shape[1])
        smooth_pently = self.smoother_kernel_1d(mask_for_smooth_pently.to(torch.float)).cpu().abs().sum(dim = (1, 2)).detach().numpy()
        smooth_pently = smooth_pently.reshape(mask.shape[0], -1).sum(axis = 1)/(2*mask.shape[1] * mask.shape[2])
                
        sparse_pently = mask.shape[1] * mask.shape[2] - mask.sum(dim = (1, 2)).cpu().detach().numpy()
        
        x_o = input * mask

        
        if self.max_batch_size == None:
            self.get_max_batch_size(x_o, mask)


        self.genmodel.train()
        marginalisation = []
        for _ in range(5):
            prob = []
            for i in range(0, x_o.shape[0], self.max_batch_size):
                perturbed_samples = self.genmodel.sample(x_o[i:i+self.max_batch_size], mask[i:i+self.max_batch_size])
                prob.append(self.net.prob(perturbed_samples))
            prob = torch.cat(prob, dim = 0).unsqueeze(0)
            marginalisation.append(prob)
        marginalisation = torch.cat(marginalisation, dim = 0).mean(0)
        marginalisation_label = torch.argmax(marginalisation, dim = 1)


        if self.counter_class == None:
            labelchange = np.array([0 if marginalisation_label[i] != self.origin_pred_label else 1 for i in range(len(marginalisation_label))])
        else:
            labelchange = np.array([0 if marginalisation_label[i] == self.counter_class else 1 for i in range(len(marginalisation_label))])
        
        return sparse_pently * 10 + smooth_pently + labelchange * 1e8

    
    def attribute_BPSO(self, input, **kwargs):
        step = 1
        best_cost = float('inf')
        costs = []
        while True:
            c1 = 5
            c2 = 5
            w = 1
            iters = 5000
            early_stop = 20
            n_particles = 100
            options = {'c1': c1 , 'c2': c2, 'w': w, 'k': n_particles, 'p':2}
            
            if step != 1:
                init_pos = np.ones((n_particles, len(self.blocks)))
                init_pos = torch.bernoulli(torch.tensor(init_pos).to(torch.float)).detach().numpy()
            else:
                init_pos = None
            
            optimizer = BPSO_ES(n_particles = n_particles, dimensions=len(self.blocks),init_pos = init_pos, options= options)
            cost, pos = optimizer.optimize(self.fitnessfun, iters=iters, input = input, early_stop = early_stop)
            
            step += 1
            best_selected_blocks = np.nonzero(pos)[0]

            if cost < best_cost:
                attrs = self.block_to_mask(best_selected_blocks)
                best_cost = cost

            if cost >= 1e8:
                return 1 - attrs, costs
            
            costs += optimizer.cost_history
            
            if self.block_max_size() < 2:
                break
            self.blocks = self.next_meshing(self.blocks, best_selected_blocks)

        return 1 - attrs, costs

    def attribute_GA_parallel(self, input, **kwargs):
    
        def fitness_func(solution):
            return self.fitnessfun(solution, input)

        best_cost = float('inf')

        costs = []
        step = 0
        while True:

            sol_per_pop = 100
            num_genes = len(self.blocks)

            algorithm_param = {'max_num_iteration': 1000,\
                   'population_size':100,\
                   'mutation_probability':0.2,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.5,\
                   'crossover_type':'one_point',\
                   'max_iteration_without_improv':20}

            if step != 0:
                init_pos = np.ones((sol_per_pop, num_genes), dtype = int)
            else:
                init_pos = None

            ga_instance = GA_parallel(function = fitness_func, dimension = len(self.blocks), algorithm_parameters = algorithm_param,  convergence_curve=False)
            ga_instance.run(init_pos = init_pos)

            best_soluction, cost = ga_instance.best_variable, ga_instance.best_function   

            best_selected_blocks = np.nonzero(best_soluction)[0]

            if cost < best_cost:
                attrs = self.block_to_mask(best_selected_blocks)
                best_cost = cost
            
            if cost >= 1e8:
                return 1- attrs, costs
            costs += ga_instance.best_history
            
            if self.block_max_size() < 2:
                break
        
            self.blocks = self.next_meshing(self.blocks, best_selected_blocks)
            step += 1
        return 1 - attrs, costs


    def attribute(self, input, target, **kwargs):
        '''
        counter_class -> target for the counterfactual samples

        '''
        self.genmodel = kwargs['GenModel']
        init_scaler = kwargs['init_scaler']
        self.input_dim = input.shape[-2:]
        self.counter_class = None if 'counter_class' not in kwargs.keys() else kwargs['counter_class']
        self.origin_pred_label = target

        self.blocks = self.meshing(Block(start_point = (0, 0), end_point = self.input_dim), scaler = init_scaler, block_start_index = 0)

        if kwargs['SearchMethod'] == 'BPSO':
            return self.attribute_BPSO(input)
        
        if kwargs['SearchMethod'] == 'GA_parallel':
            return self.attribute_GA_parallel(input)
