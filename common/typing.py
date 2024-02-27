from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import jax.numpy as jnp
import flax

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]
Array = Union[np.ndarray, jnp.ndarray]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
ModuleMethod = Union[
    str, Callable, None
]  # A method to be passed into TrainState.__call__
