from transformer_lens import HookedTransformer
from transformers import PreTrainedModel
import torch
import einops

def orthogonalize_by_projection_matrix(weight: torch.Tensor, projection_matrix: torch.Tensor):
    """
    Orthogonalize the weight tensor (shape [out, in]) using the projection matrix (shape [out, out]).
    (Literally just post-multiply but with some reshaping.)
    """
    return torch.mm(projection_matrix, weight)

def orthogonalize(weight: torch.Tensor, vector: torch.Tensor):
    """
    Orthogonalize the weight tensor (shape [out, in]) wrt the vector (shape [out]).
    """
    # W' = W - vv^T W

    projection = torch.eye(vector.shape[0]) - vector.unsqueeze(1) @ vector.unsqueeze(0)

    return orthogonalize_by_projection_matrix(weight, projection)

def generate_weight_order(
    model_cfg,
    orthogonalize_cfg = {
        "embeddings": True,
        "mlp_in": False,
        "mlp_out": True,
        "attn_qkv": False,
        "attn_out": True,
    }
):
    weight_order = ["model.embed_tokens"] if orthogonalize_cfg["embeddings"] else []

    for layer_idx in range(model_cfg.num_hidden_layers):
        if orthogonalize_cfg["mlp_in"]:
            # probably requires standard MLP (i.e. no gated MLP things)
            # weight_order.append(("blocks.{layer_idx}.mlp.W_in", f"blocks.{layer_idx}.mlp.hook_pre"))
            raise NotImplementedError("mlp_in")

        if orthogonalize_cfg["mlp_out"]:
            weight_order.append(f"model.layers.{layer_idx}.mlp.down_proj")

        if orthogonalize_cfg["attn_qkv"]:
            raise NotImplementedError("attn_qkv")

        if orthogonalize_cfg["attn_out"]:
            weight_order.append(f"model.layers.{layer_idx}.self_attn.o_proj")

    return weight_order

class OrthogonalizedTransformer:
    def __init__(self, model):
        self.model = model
    
    def orthogonalize_weight(self, key, projection_or_vector):
        """
        Replace a weight with it's orthogonalized version.
        The key should be the name of a `torch.nn.Linear` module.
        """
        parameter = self.model.get_parameter(f"{key}.weight")
        weight = parameter.data
        initial_shape = weight.shape

        is_attn = "attn" in key
        is_embed = "embed" in key
        is_mlp = "mlp" in key

        # if is_attn:
        #     # if weight is an attention weight, it has shape [n_heads, d_head, d_model]
        #     # therefore, first reshape to [d_model, n_heads * d_head] = [d_model, d_model]
        #     # do the orthogonalization, and then reshape back
        #     print(weight.shape)

        #     N_HEADS, D_HEAD, D_MODEL = weight.shape
        #     weight = einops.rearrange(
        #         weight, "n_heads d_head d_model -> d_model (n_heads d_head)",
        #         n_heads=N_HEADS, d_head=D_HEAD, d_model=D_MODEL
        #     )
        
        # if is_embed or is_mlp:
        #     weight = weight.T

        if is_embed:
            weight = weight.T

        is_vector = len(projection_or_vector.shape) == 1

        if is_vector:
            orthogonalized_weight = orthogonalize(weight, projection_or_vector)
        else:
            orthogonalized_weight = orthogonalize_by_projection_matrix(weight, projection_or_vector)
        
        # if is_attn:
        #     orthogonalized_weight = einops.rearrange(
        #         orthogonalized_weight, "d_model (n_heads d_head) -> n_heads d_head d_model",
        #         n_heads=N_HEADS, d_head=D_HEAD, d_model=D_MODEL
        #     )
        
        # if is_embed or is_mlp:
        #     orthogonalized_weight = orthogonalized_weight.T

        if is_embed:
            orthogonalized_weight = orthogonalized_weight.T

        assert initial_shape == orthogonalized_weight.shape

        parameter.data = orthogonalized_weight.contiguous()

    def set_projection_hooks(self, key, projection_or_vector_or_none = None):
        """
        Replace the hook on the weight with a projection matrix.
        The key should be a valid hook point.
        """

        # unset the current hook

        self.model.get_submodule(key).remove_hooks()

        if projection_or_vector_or_none is None:
            return

        is_vector = len(projection_or_vector_or_none.shape) == 1

        if is_vector:
            proj_matrix = torch.eye(projection_or_vector_or_none.shape[0]) - projection_or_vector_or_none.unsqueeze(1) @ projection_or_vector_or_none.unsqueeze(0)
        else:
            proj_matrix = projection_or_vector_or_none

        self.model.get_submodule(key).add_hook(
            lambda act, _: torch.einsum("... j, k j -> ... k", act, proj_matrix)
        )
    
    def get_activations(self, module_to_hook, *args, **kwargs):
        cache = None

        def write_to_cache_hook(module, args, output):
            nonlocal cache
            cache = output
        
        handle = self.model.get_submodule(module_to_hook).register_forward_hook(write_to_cache_hook)
        self.model.forward(*args, **kwargs)
        handle.remove()

        return cache
