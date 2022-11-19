"""optim.py.
"""

import torch

AVAI_OPTIMS = ["adam", "sgd", "adamw"]

# Config for Adam and AdamW
ADAM_BETAS = (0.9, 0.999)

# Config for SGD
MOMENTUM = 0.9
SGD_NESTEROV = False

def build_optimizer(params_groups, name, lr, weight_decay):
    """Build optimizer.
    Args:
        params_groups (list[dict]): list of parameters groups.
        name (str): name of optimizer.
        **kwargs: other arguments.
    """
    assert name in AVAI_OPTIMS, f"Optimizer {name} not found; available optimizers = {AVAI_OPTIMS}"
    if name == "sgd":
        return build_sgd_optimizer(params_groups, lr, weight_decay)
    elif name == "adam":
        return build_adam_optimizer(params_groups, lr, weight_decay, betas=ADAM_BETAS)
    elif name == "adamw":
        return build_adamw_optimizer(params_groups, lr, weight_decay, betas=ADAM_BETAS)
    else:
        raise ValueError("Unknown optimizer: {}".format(name))
    

def build_sgd_optimizer(params_groups, lr, weight_decay, momentum=MOMENTUM, nesterov=SGD_NESTEROV):
    """Build SGD optimizer.
    Args:
        params_groups (list[dict]): list of parameters groups.
        lr (float): learning rate.
        momentum (float): momentum.
        weight_decay (float): weight decay.
        nesterov (bool): whether to use nesterov.
    """
    return torch.optim.SGD(
        params_groups,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )

def build_adam_optimizer(params_groups, lr, weight_decay, betas=ADAM_BETAS):
    """Build Adam optimizer.
    Args:
        params_groups (list[dict]): list of parameters groups.
        lr (float): learning rate.
        betas (tuple[float]): coefficients used for computing running averages of gradient and its square.
        weight_decay (float): weight decay.
    """
    return torch.optim.Adam(
        params_groups, lr=lr, weight_decay=weight_decay, betas=betas
    )

def build_adamw_optimizer(params_groups, lr, weight_decay, betas=ADAM_BETAS):
    """Build AdamW optimizer.
    Args:
        params_groups (list[dict]): list of parameters groups.
        lr (float): learning rate.
        betas (tuple[float]): coefficients used for computing running averages of gradient and its square.
        weight_decay (float): weight decay.
    """
    return torch.optim.AdamW(
        params_groups, lr=lr, weight_decay=weight_decay, betas=betas
    )