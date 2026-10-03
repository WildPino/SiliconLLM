"""Scoped independent I64 projection for ALL serialized I8 matrix inputs."""
from contextlib import contextmanager
import meth336_switch_integer_reference as R
import meth345_switch_head_a16_reference as H

projection = H.projection
target_control_model = R.target_control_model


@contextmanager
def compact_reference(model, entries, payload):
    # The existing integer-reference closures resolve the projection at call
    # time. Only this sequential reference context installs A16 for ALL I8
    # Linear modules; F32 routers and rounded attention/norm stay qualified.
    previous = R.integer_projection
    R.integer_projection = projection
    try:
        with R.compact_reference(model, entries, payload) as mode:
            yield mode
    finally:
        R.integer_projection = previous
