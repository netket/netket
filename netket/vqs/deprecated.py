from netket.utils import deprecated_new_name
from .full_summ import FullSumState


@deprecated_new_name(
    "nk.vqs.FullSumState",
    reason="""
    ExactState has been renamed to FullSumState to better
    reflect its purpose.
    """,
)
def ExactState(*args, **kwargs):
    return FullSumState(*args, **kwargs)
