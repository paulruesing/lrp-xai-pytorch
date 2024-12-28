from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

import torch
from torch import nn
import torchvision

from src.xai.lrp_utils import layers_lookup
import src.utils.plotting as plotting


DEBUG = False


class LRPEngine:
    def __init__(self,
                 model: torch.nn.Module,
                 input_batch: torch.Tensor=None,
                 plot_output_dir=None,
                 classifier_spatial_dim=None
                 ):
        """
        To be commented.
        :param model:
        :param input_batch: torch.Tensor, default=None:
            input batch as 4D tensor (unsqueezed), can be provided later but is required for calculations.
        :param plot_output_dir: string, default=None:
            describes the output folder
        :param classifier_spatial_dim: int, default=None:
            spatial input dimension to classifier layers. if not provided will be inferred from last pooling layer.
        """
        self.model = model
        self._input_batch = input_batch  # as property for distinct setter behavior
        self.plot_output_dir = plot_output_dir

        # input dimension to classifier layers:
        if classifier_spatial_dim is None:
            pooling_layers = [l for l in model.children() if
                              isinstance(l, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.AdaptiveMaxPool2d, nn.MaxPool2d))]
            try:
                self.classifier_spatial_dim = pooling_layers[-1].output_size[0]  # infer from last pooling layer
            except BaseException:
                raise AttributeError("Please provide classifier spatial dimension manually! Inferring from last pooling layer did not work.")
        else:
            self.classifier_spatial_dim = classifier_spatial_dim

        # private attributes readable through properties:
        self._layers = None
        self._activations = None
        self._relevance_scores = None
        self._probabilities = None

    @property
    def input_batch(self):
        """ Input data, must be a 4D (unsqueezed) tensor. """
        if self._input_batch is None:
            raise AttributeError("Please provide input data (input_batch)!")
        return self._input_batch

    @input_batch.setter
    def input_batch(self, value):
        """ Mutable but resets results. """
        self._input_batch = value
        self._probabilities = None
        self._activations = None
        self._relevance_scores = None

    @property
    def probabilities(self):
        if self._probabilities is None:
            self._probabilities = torch.nn.functional.softmax(self.activations[-1].detach().view(-1), dim=0)
        return self._probabilities

    @property
    def activations(self):
        """
        If not already done, executes the prepared model and collects the activation tensors in a list.
        :return:
            a list of torch.Tensor objects containing the neuron activations
        """
        if self._activations is None:
            # initial activation is input batch
            A = [self.input_batch] + [None] * len(self.layers)
            # forward pass (without torch.no_grad() because we will later call backward)
            for layer_ind, layer in enumerate(self.layers):
                try:
                    A[layer_ind + 1] = layer.forward(A[layer_ind])
                except RuntimeError as err:
                    raise BaseException(
                        f"Couldn't compute layer {layer_ind}: {self.layers[layer_ind]}! Error message:\n{err}")
            self._activations = A
        return self._activations

    @property
    def layers(self):
        """
        Executes auxiliary method that converts model into optimal ModuleList for LRP calculation.
        :return: list of torch.nn.Module: list of layers
        """
        if self._layers is None:
            self._load_model_lrp()
        return self._layers

    def print_results(self, categories, len_list=3):
        # show top categories per image (torch.topk returns values and index)
        top_probs, top_class_ids = torch.topk(self.probabilities, len_list)
        print("Classification results:")
        print("------------------------------------------------------------------------")
        for i in range(top_probs.size(0)):
            print(
                f"'{categories[top_class_ids[i]]}' with prob. {top_probs[i].item()} (category no. {top_class_ids[i].item()})")
        print("------------------------------------------------------------------------")

    @property
    def relevance_scores(self):
        if self._relevance_scores is None:
            raise BaseException("Relevance scores need to be calculated through calculate_relevance_scores_resnet() or calculate_relevance_scores_vgg() first!")
        return self._relevance_scores

    def calculate_relevance_scores(self, rel_filter_ratio : float = 1.0, **lrp_kwargs):
        """
        This method calculates relevance propagation scores utilising multiple LRP rules from the LRPModel class.

        :param rel_filter_ratio: float, default 1.0:
            if <1.0 a relevance filter is used, that only lets the highest rel_filter_ratio*100 % LRP values propagate
            through each calculation step. Should yield more expressive outputs if used!
        :param lrp_kwargs:
            kwargs_dict for LRPModel initialisation
        :return:
        """
        lrp_model = LRPModel(module_list=self.layers, rel_pass_ratio=rel_filter_ratio, **lrp_kwargs)
        print('Initialised LRP model. Now calculating relevance...')
        lrp_model.forward(self.input_batch, topk=1)
        self._relevance_scores = lrp_model.relevance_scores

    @staticmethod
    def unpack_layer_container(container, list_of_layers: nn.ModuleList = None):
        """
        Recursive method to unpack layer containers (suitable for ResNet and VGG backbone).

        :param container: model or sequential container to unpack
        :param list_of_layers: list of layers to append layers to (then not returning anything), for recursive calling

        :return: list_of_layers if such was not provided as an argument
        """
        # initialise list if initial call and set inplace to False, meaning return something:
        if list_of_layers is None:
            list_of_layers = nn.ModuleList()
            inplace = False
        else:
            inplace = True  # do not return anything, just append to given list

        # if container is not single layer:
        if isinstance(container, (torch.nn.Sequential, torchvision.models.resnet.ResNet, torchvision.models.VGG)):
            # iterate through container elements:
            for container_name, layers in container.named_children():
                # if we face another container:
                if isinstance(layers, torch.nn.Sequential):
                    LRPEngine.unpack_layer_container(layers, list_of_layers)
                # otherwise append elements:
                else:
                    list_of_layers.append(layers)

        # if container is single layer or Bottleneck module just append such:
        else:
            list_of_layers.append(container)

        # if no list provided (initial call of recursive method) then return list
        if not inplace:
            return list_of_layers

    def _load_model_lrp(self):
        """
        This method converts an PyTorch model implementation into an ideal ModuleList suitable for LRP calculation.
        Specifically, it converts linear layers into equivalent convolutional ones to facilitate relevance propagation.
        """
        # unpack all containers into a list of layers:
        layer_list = self.unpack_layer_container(self.model)
        linear_counter = 0  # requires index for distinctive handling of linear layers depending on respective input

        # iterate through layers to convert linear ones into equivalent convolutional ones:
        for layer_index, layer in enumerate(layer_list):
            # convert linear layers:
            if isinstance(layer, torch.nn.Linear):
                # distinctive handling of first linear layer, because such follows a flattened output
                in_channels = layer.in_features if linear_counter > 0 else int(
                    layer.in_features / (self.classifier_spatial_dim ** 2))
                spatial_dim = 1 if linear_counter > 0 else self.classifier_spatial_dim
                print(f"Converting linear layer ({layer}) into equivalent convolutional one with {in_channels} input channels and {spatial_dim} kernel size.")
                # save new layer:
                layer_list[layer_index] = self._convert_linear_to_conv(layer, in_channels, spatial_dim)
                linear_counter += 1

        # remove flatten layers (because these are only necessary for linear layers which were converted)
        layer_ind = len(layer_list)-1
        for layer in layer_list[::-1]:  # iterate in reverse order to prevent issues with wrong indices
            if isinstance(layer, torch.nn.Flatten):
                print(f'Removing flattening layer {layer_ind}.')
                layer_list.pop(layer_ind)
            layer_ind -= 1  # needs to be done manually because enumerate doesn't yield true index with reversed list

        # set ModuleList into evaluation mode and save
        layer_list.eval()
        self._layers = layer_list

    def plot_relevance_scores(self, layer_index=0, plt_cmap="afmhot", input_reference=None, hidden=False) -> None:
        """
        Plots layer-wise relevance propagation score heatmap next to original image.
        LRP scores get min-max scaled, meaning the lowest values become 0 the highest 1.
        Colormap can be changed with plt_cmap parameter, default is "afmhot".

        This method's is inspired by https://github.com/kaifishr/PyTorchRelevancePropagation

        :param layer_index: int, default=0:
            index of layer to plot relevance scores for.
        :param plt_cmap: string, default="afmhot"
            matplotlib color map to use for LRP heatmap
        """
        # read out classification for titling:
        top_probs, top_class_ids = torch.topk(self.probabilities, k=1)
        class_id, prob = top_class_ids[0].item() + 1, round(top_probs[0].item(), 2)
        input_title = f" {input_reference} " if input_reference is not None else " "

        # initialise plots:
        colorbar_size = 0.05   # defines ratio of width of colorbar related to heatmap:
        # we define the right subplot a little bit wider based on the colorbar_size and padding:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), width_ratios=[1, 1 + colorbar_size + 0.012])

        # print original image:
        x = self.input_batch[0].squeeze().permute(1, 2, 0).detach().cpu()
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max - x_min)
        axes[0].imshow(x)
        axes[0].set_axis_off()
        axes[0].set_title(f"Input Image{input_title}(Class {class_id} with {prob})")

        # print relevance scores:
        r = self.relevance_scores[layer_index][0].sum(axis=0)
        r_min = r.min()
        r_max = r.max()
        r = (r - r_min) / (r_max - r_min)
        # plot heatmap with colorbar:
        try:
            color_object = axes[1].imshow(r, cmap=plt_cmap)
        except ValueError as err:
            print(err)
            print("Therefore, now using default value 'afmhot' instead.")
            color_object = axes[1].imshow(r, cmap="afmhot")
        axes[1].set_axis_off()
        axes[1].set_title(f"LRP Heatmap (Layer {layer_index})")

        # color bar:
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size=f"{colorbar_size * 100}%", pad=0.12)
        fig.colorbar(color_object, cax=cax, ticks=[1, 0],
                     format=mticker.FixedFormatter(['Strengthen', 'Weaken']),)
        cax.set_title('Colormap')

        fig.tight_layout()
        # save plot if output directory for plots is defined:
        if self.plot_output_dir is not None:
            filename = plotting.file_title(title=f'LRP Calc{input_title}class {class_id} prob {prob} layer {layer_index}',
                                           dtype_suffix='.png')
            plt.savefig(os.path.join(self.plot_output_dir, filename))

        # show plot:
        if not hidden:
            plt.show()

    ############################# Auxiliary Static Methods #############################
    @staticmethod
    def _convert_linear_to_conv(linear_layer, in_channels=512, spatial_dim=7):
        '''
        Converts a linear layer to an equivalent convolutional layer.

        :param linear_layer: The linear layer to be converted
        :param in_channels: Number of input channels to the original convolutional feature map
        :param spatial_dim: Spatial dimensions of the input feature map (assumed square)

        :return: An equivalent convolutional layer
        '''

        # Calculate kernel size to match the spatial dimensions
        kernel_size = spatial_dim  # This should be 7 in this case

        # Create a new Conv2d layer with equivalent parameters
        conv_layer = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=linear_layer.out_features,
            kernel_size=kernel_size
        )

        # Reshape the linear weights to fit Conv2d weights:
        # The shape should be [out_features, in_channels, kernel_size, kernel_size]
        conv_layer.weight.data = linear_layer.weight.data.view(
            linear_layer.out_features,
            in_channels,
            kernel_size,
            kernel_size
        )

        # Copy the bias from the linear layer if it exists
        if linear_layer.bias is not None:
            conv_layer.bias.data = linear_layer.bias.data.clone()

        return conv_layer


############################# LRP Model Class #############################
# The code in this section is an amended but based on
# https://github.com/keio-smilab24/LRP-for-ResNet/tree/main?tab=License-1-ov-file#readme.
class LRPModel(nn.Module):
    """
    Class wraps PyTorch model to perform layer-wise relevance propagation.
    Capable of residual networks!
    """
    def __init__(self, module_list: torch.nn.ModuleList, rel_pass_ratio: float = 0.0, skip_connection_prop="latest") -> None:
        super().__init__()
        self.layers = module_list
        self.rel_pass_ratio = rel_pass_ratio
        self.skip_connection_prop = skip_connection_prop
        self._relevance_scores = None  # list to save relevance scores

        # Create LRP network
        self.lrp_layers = self._create_lrp_model()

    def _create_lrp_model(self) -> torch.nn.ModuleList:
        """
        Method builds the model for layer-wise relevance propagation.

        Returns:
            LRP-model as module list.
        """
        # Clone layers from original model. This is necessary as we might modify the weights.
        layers = deepcopy(self.layers)
        lookup_table = layers_lookup(self.skip_connection_prop)

        # Run backwards through layers
        for i, layer in enumerate(layers[::-1]):
            try:
                layers[i] = lookup_table[layer.__class__](layer=layer, top_k=self.rel_pass_ratio)
            except KeyError:
                message = (
                    f"Layer-wise relevance propagation not implemented for "
                    f"{layer.__class__.__name__} layer."
                )
                raise NotImplementedError(message)

        return layers

    @property
    def relevance_scores(self):
        if self._relevance_scores is None:
            raise RuntimeError("Relevance scores first need to be calculated through forward(input)")
        return self._relevance_scores

    def forward(self, x: torch.tensor, topk=-1) -> torch.tensor:
        """
        Forward method that first performs standard inference followed by layer-wise relevance propagation.

        Args:
            x: Input tensor representing an image / images (N, C, H, W).

        Returns:
            Tensor holding relevance scores with dimensions (N, 1, H, W).
        """
        activations = list()
        relevance_list = list()

        # Run inference and collect activations.
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            activations.append(torch.ones_like(x))
            for layer in self.layers:
                x = layer.forward(x)
                activations.append(x)

        # Reverse order of activations to run backwards through model
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]

        # Initial relevance scores are the network's output activations
        relevance = torch.softmax(activations.pop(0), dim=-1)  # Unsupervised
        # topk decides how many output probabilities (counting from the highest downwards) should be utilized for
        # the relevance of the output layer, i.e. should be considered for LRP calculation
        # -1 means consider all
        if topk != -1:
            relevance_zero = torch.zeros_like(relevance)
            top_k_indices = torch.topk(relevance, topk).indices
            for index in top_k_indices:
                relevance_zero[..., index] = relevance[..., index]
            relevance = relevance_zero

        # Perform relevance propagation
        for i, layer in enumerate(self.lrp_layers):
            a = activations.pop(0)
            try:
                # store relevance of interim layers:
                relevance_list.append(relevance.clone().detach().cpu())
                # propagate relevance:
                relevance = layer.forward(a, relevance)
            except RuntimeError:
                print(f"RuntimeError at layer {i}.\n"
                      f"Layer: {layer.__class__.__name__}\n"
                      f"Relevance shape: {relevance.shape}\n"
                      f"Activation shape: {activations[0].shape}\n")
                exit(1)

        # append last relevance
        relevance_list.append(relevance)

        # reverse list because we propagated backwards
        self._relevance_scores = relevance_list[::-1]