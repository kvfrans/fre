###############################
#
#  Some shared utility functions
#
###############################

import jax

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wraps a function to supply jax rng. It will remember the rng state for that function.
    """
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped

