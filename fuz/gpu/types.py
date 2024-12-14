"""Types and constants for fuz."""

from jaxtyping import Array as JArr, Real as JReal



JaxScalar = JReal[JArr, ''] | JReal[JArr, '1']
JaxVec = JReal[JArr, 'dim']
JaxTensor = JReal[JArr, '...']