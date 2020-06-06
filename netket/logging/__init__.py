from ._json_log import JsonLog

from ..utils import tensorboard_available

if tensorboard_available:
    from ._tensorboard import TBLog
