from cb_utils.models import DEVICE, load_demo_gpt2
import tqdm
from tasks.task import Task
import torch
from collections import defaultdict

def clamp_weights(param_dict, edge_threshold=0.5, weight_threshold=0.5):
    """
    Clamp both edge and weight masks to be either 0 or 1. param_dict should only have edge and weight masks that you want to clamp.
    """
    for name, p in param_dict.items():
        if "edge_mask" in name:
            p.data = torch.where(p.data < edge_threshold, torch.zeros_like(p.data), torch.ones_like(p.data))
        elif "weight_mask" in name:
            p.data = torch.where(p.data < weight_threshold, torch.zeros_like(p.data), torch.ones_like(p.data))


def evaluate_model(model, eval_tasks: dict[str, Task], num_eval_steps: int=1):
    """
    Evaluate a model on a set of Tasks. Returns a dictionary of task names to losses.
    """
    losses = {}
    with torch.no_grad():
        for task_name, task in eval_tasks.items():
            print(f"Evaluating on {task_name}")
            losses[task_name] = 0
            for i in range(num_eval_steps):
                losses[task_name] += task.get_test_loss(model)
            losses[task_name] /= num_eval_steps
            print(f"Loss on {task_name}: {losses[task_name]}")
    return losses


def train_masks(model, tasks: dict[str, Task],
               optimizer: torch.optim.Optimizer,
               mask_params: dict[str, torch.nn.parameter.Parameter],
               task_weights: dict[str, float],
               num_epochs: int, 
               eval_tasks: dict[str, Task]=None,
               steps_per_epoch=100,
               evaluate_every=10, 
               clamp_every=50, 
               threshold=0.5, 
               edge_mask_reg_strength=None | float | function, 
               weight_mask_reg_strength=None | float | function,
               num_eval_steps=1
               ):
    """
    Train a model using tasks (weight the overall loss by task_weights). For now, planned to be training differentiable binary masks over the weights and edges of the model.
    
    Parameters:
    model: DemoTransformer, the model to use for training and evaluation. If edge or weight mask should be frozen, do this to model before passing it in.
    optimizer: torch.optim.Optimizer, the optimizer to use for training. For now, planned to be over edge mask and weight mask parameters.
    mask_params: dictionary of torch.nn.parameter.Parameter with names for keys, the parameters to optimize over. Should be edge masks and weight masks. Used for clamping and saving weights.

    tasks: dictionary of Tasks with names for keys, the tasks to train on
    task_weights: dictionary floats, the weights to use for each task
    num_epochs: int, the number of epochs to train for

    eval_tasks: either None or a dictionary of tasks with task names. If none, evaluate on the training tasks.
    steps_per_epoch: int, the maximum number of steps to train for each epoch
    evaluate_every: int, the number of steps between evaluations
    clamp_every: int, the number of steps between "clamping" weights. In this context, means setting all weights below a threshold to 0 and all weights above a threshold to 1.
    threshold: float, the threshold to use for clamping weights

    edge_mask_reg_strength: float or function of epoch, the strength of the regularization on the edge masks. If None, no regularization on edge mask is used. Should be None if edge masks not being optimized.
    weight_mask_reg_strength: float or function of epoch, the strength of the regularization on the weight masks. If None, no regularization on weight mask is used. Should be None if weight masks not being optimized or if weight masks are being optimized in optimizer with some other regularization.

    """
    if eval_tasks is None:
        eval_tasks = tasks
    
    # model = load_demo_gpt2(means=means, weight_masks_attn=weight_masks_attn, weight_masks_mlp=weight_masks_mlp)
    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    for epoch in tqdm(range(num_epochs)):
        for step in range(steps_per_epoch):
            model.train()
            model.zero_grad()
            total_loss = 0
            for task_name, task in tasks.items():
                loss = task.get_train_loss(model)
                # add item (without gradients to avoid memory leak) to train_losses
                train_losses[task_name].append((epoch, step, loss.item()))
                total_loss += loss * task_weights[task_name]
            
            # Add regularization losses for edge and weight masks, l1
            
            edge_reg_term = 0
            weight_reg_term = 0
            for name, p in mask_params.items():
                if "edge_mask" in name:
                    # get l1 norm of edge mask
                    edge_reg_term += p.abs().sum()

                elif "weight_mask" in name:
                    weight_reg_term += p.abs().sum()
                
            if edge_mask_reg_strength is not None:
                if callable(edge_mask_reg_strength):
                    edge_reg_term = edge_mask_reg_strength(epoch)
                else:
                    edge_reg_term = edge_mask_reg_strength

                train_losses['edge_mask_reg'].append((epoch, step, edge_reg_term.item()))
                total_loss += edge_reg_term * edge_mask_reg_strength

            if weight_mask_reg_strength is not None:
                if callable(weight_mask_reg_strength):
                    weight_reg_term = weight_mask_reg_strength(epoch)
                else:
                    weight_reg_term = weight_mask_reg_strength

                train_losses['weight_mask_reg'].append((epoch, step, weight_reg_term.item()))
                total_loss += weight_reg_term * weight_mask_reg_strength
            

            train_losses['total'].append((epoch, step, total_loss.item()))
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if step % evaluate_every == 0:
                print(f"Epoch {epoch}, step {step}: train loss {total_loss}")
                model.eval()
                test_losses = evaluate_model(model, eval_tasks, num_eval_steps)
                for task_name, task in eval_tasks.items():
                    test_losses[task_name].append((epoch, step, test_losses[task_name]))

            if step % clamp_every == 0:
                print(f"Clamping weights")
                clamp_weights(mask_params, edge_threshold=threshold, weight_threshold=threshold)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
