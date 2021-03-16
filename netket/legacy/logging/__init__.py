from ._json_log import JsonLog

from netket.utils import tensorboard_available

if tensorboard_available:
    from ._tensorboard import TBLog
