"""Types and constants for fuz."""

from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, TypeAlias, TypeVar

from attrs import frozen
from jaxtyping import Array as JArr, Real as JReal
from narwhals.typing import IntoFrameT
from numpy import ndarray as NPArr
from scipy.stats._distn_infrastructure import rv_continuous_frozen

# Constants
RATE_N = 2

# Setup
Arr = NPArr  # TODO(viamiraia): modify in the future after creating GPU jax support


class FuzDist(Protocol): ...


# %% Basics
ArrScalar = JReal[Arr, ''] | JReal[Arr, '1']
ArrVec = JReal[Arr, 'dim']
ArrTensor = JReal[Arr, '...']
NPScalar = JReal[NPArr, ''] | JReal[NPArr, '1']
NPVec = JReal[NPArr, 'dim']
NPTensor = JReal[NPArr, '...']
JaxScalar = JReal[JArr, ''] | JReal[JArr, '1']
JaxVec = JReal[JArr, 'dim']
JaxTensor = JReal[JArr, '...']

# %%% General
Scalar = float | int | ArrScalar
VecLike = ArrVec | Sequence[Scalar]
Broadcast = Scalar | ArrTensor
BroadcastT = TypeVar('BroadcastT', bound=Broadcast)

# %% Distributions

# %%% Beta distribution
ABTuple = tuple[Scalar, Scalar]
ABLocScale = ABTuple | tuple[Scalar, Scalar, Scalar] | tuple[Scalar, Scalar, Scalar, Scalar]
ABBroadcast = tuple[Broadcast, Broadcast]

# %% Ranking-specific
ArrPair = JReal[Arr, '2'] | JReal[Arr, '2 1'] | JReal[Arr, '1 2']
ArrPairs = JReal[Arr, '2 dim'] | JReal[Arr, 'dim 2']
SeqPair = Annotated[Sequence[Scalar], 2]
RealPair = Annotated[Sequence[Real], 2]
RatingPair = RealPair | ArrPair
Rankable = IntoFrameT | ArrPairs | Sequence[RatingPair] | Mapping[str, RatingPair]
Ranked = IntoFrameT | tuple[ArrPairs, ArrVec] | tuple[Mapping[str, RatingPair], tuple[str]]
Columns = Literal['col', 'column', 'cols', 'columns', 1]
Rows = Literal['row', 'rows', 0]
Auto = Literal['auto', 'automatic']
XVec = JReal[Arr, 'x'] | JReal[Arr, '1 x']
DistVec = JReal[Arr, 'dist'] | JReal[Arr, 'dist 1']
DistMat = JReal[Arr, 'dist x']
PairLike = ArrPair | SeqPair

# %% Utils
ListOfDicts = list[dict[str, Any]]
DictOfLists = dict[str, list[Any]]

# %% Avoiding circular imports

# %% Pooling
# Old Type Aliases
TRealVec = NPVec | Sequence[Real]
SampRvs = Sequence[NPVec]
ContDist = FuzDist | rv_continuous_frozen
ContDists = Sequence[ContDist]
PoolNumF = Callable[[VecLike], NPVec]
PoolF = Callable[[VecLike], NPVec]
PoolNumFS = Callable[[], NPVec]
TPoolFS = Callable[[NPVec], NPVec]
