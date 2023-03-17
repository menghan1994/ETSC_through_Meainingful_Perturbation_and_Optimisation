from captum.attr import IntegratedGradients, Saliency, Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression

import torch
from torch.nn import CosineSimilarity
from captum._utils.common import _flatten_tensor_or_tuple

from explainer.HeuristicalSearchBase import SegmentIdentificationSearch

def saliency(net, input, target, methods, GenModel=None, **kwargs):
    net.train()
    input.requires_grad = True
    
    if methods == 'Saliency':
        saliency = Saliency(net)
        attr = saliency.attribute(input, target)
        attr = (attr.cpu().detach().numpy(), 0)
    
    if methods == 'IntegratedGradients':
        saliency = IntegratedGradients(net)
        attr = saliency.attribute(input, target = target)
        attr = (attr.cpu().detach().numpy(), 0)
    
    if methods == "LIME":
        lime = Lime(net, SkLearnLinearRegression())
        attr = lime.attribute(input, target=target, n_samples=2000)
        attr = attr.cpu().detach().numpy().squeeze()
        attr = (attr, 0)

    if methods == "LIME-G":

        def similarity_kernel(original_input, perturbed_input, perturbed_interpretable_input, **kwargs):
            kernel_width = 1.0
            # l2_dist = torch.norm(original_input - perturbed_input)
            flattened_original_inp = _flatten_tensor_or_tuple(original_input).float()
            flattened_perturbed_inp = _flatten_tensor_or_tuple(perturbed_input).float()
            cos_sim = CosineSimilarity(dim=0)
            distance = 1 - cos_sim(flattened_original_inp, flattened_perturbed_inp)
            return torch.exp(- (distance**2) / (kernel_width**2))

        def perturb_func(original_input, **kwargs):
            return torch.bernoulli(torch.ones_like(original_input) * 0.5).reshape(1, -1)

        def from_interp_rep_transform(curr_sample, original_input, **kwargs):
            curr_sample = curr_sample.reshape(original_input.shape)
            GenModel = kwargs['GenModel']
            x_o = curr_sample * original_input
            perturbation_inputs = GenModel.sample(x_o, curr_sample)
            return torch.mean(perturbation_inputs, dim = 0, keepdim = True)

        lime_G = LimeBase(net,
                        # SkLearnLinearModel("linear_model.LinearRegression", batch_size = 200),
                        SkLearnLinearRegression(),
                        similarity_func=similarity_kernel,
                        perturb_func=perturb_func,
                        perturb_interpretable_space = True,
                        from_interp_rep_transform=from_interp_rep_transform,
                        to_interp_rep_transform = None
                        )
        attr = lime_G.attribute(input, target=target, n_samples=2000, GenModel = GenModel)
        attr = attr.reshape(input.shape)
        attr = attr.detach().numpy()
        attr = (attr, 0)
    
    if methods == 'OurSearch_BPSO':
        saliency = SegmentIdentificationSearch(net, device = kwargs['device'])
        attr = saliency.attribute(input, target, GenModel = GenModel, init_scaler = kwargs['init_scaler'], SearchMethod='BPSO')
    

    if methods == 'OurSearch_GA_parallel':
        saliency = SegmentIdentificationSearch(net, device = kwargs['device'])
        attr = saliency.attribute(input, target, GenModel = GenModel, init_scaler = kwargs['init_scaler'], SearchMethod='GA_parallel')

    return attr