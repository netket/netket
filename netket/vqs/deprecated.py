from netket.utils import deprecated_new_name
from .full_summ import FullSummationState


@deprecated_new_name(
    "nk.vqs.FullSummationState",
    reason="""
    ExactState has been renamed to FullSummationState to better
    reflect its purpose.
    """,
)
def ExactState(*args, **kwargs):
    return FullSummationState(*args, **kwargs)
