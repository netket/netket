try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False


try:
    import tensorboardX

    tensorboard_available = True
except ImportError:
    tensorboard_available = False


try:
    import backpack

    backpack_available = True
except ImportError:
    backpack_available = False
