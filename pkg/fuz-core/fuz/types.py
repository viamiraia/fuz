"""Types and constants for fuz."""

from collections.abc import Callable, Mapping, Sequence
from numbers import Complex, Real
from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from jaxtyping import Num
from narwhals.typing import IntoFrameT
from numpy import ndarray as NPArr
from scipy.stats._distn_infrastructure import rv_continuous_frozen

# Constants
RATE_N = 2

# Setup
Arr = NPArr  # TODO(viamiraia): modify in the future after creating GPU jax support


@runtime_checkable
class FuzDist(Protocol): ...


@runtime_checkable
class Array(Protocol):
    """Minimal array protocol."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> Any: ...
    def __sub__(self, other: Any) -> 'Array': ...
    def __add__(self, other: Any) -> 'Array': ...
    def __mul__(self, other: Any) -> 'Array': ...
    def __truediv__(self, other: Any) -> 'Array': ...
    def __lt__(self, other: Any) -> 'Array': ...
    def __gt__(self, other: Any) -> 'Array': ...


# %% Basics
ArrScalar = Num[Any, ''] | Num[Any, '1']
ArrVec = Num[Any, 'dim']
ArrTensor = Num[Any, '...']
NPScalar = Num[NPArr, ''] | Num[NPArr, '1']
NPVec = Num[NPArr, 'dim']
NPTensor = Num[NPArr, '...']


# %%% General
# supposedly complex works better than numbers.Complex
Scalar: TypeAlias = Real | Complex
ArrScalar: TypeAlias = Scalar | Array
VecLike: TypeAlias = Sequence[Scalar] | Array
Broadcast: TypeAlias = ArrScalar | VecLike
BroadcastT = TypeVar('BroadcastT', bound=Broadcast)

# %% Distributions

# %%% Beta distribution
ABTuple = tuple[Scalar, Scalar]
ABLocScale = (
    ABTuple | tuple[Scalar, Scalar, Scalar] | tuple[Scalar, Scalar, Scalar, Scalar]
)
ABBroadcast = tuple[Broadcast, Broadcast]

# %% Ranking-specific
ArrPair = Num[Any, '2'] | Num[Any, '2 1'] | Num[Any, '1 2']
ArrPairs = Num[Any, '2 dim'] | Num[Any, 'dim 2']
SeqPair = Annotated[Sequence[Scalar], 2]
RealPair = Annotated[Sequence[Real], 2]
RatingPair = RealPair | ArrPair
Rankable = IntoFrameT | ArrPairs | Sequence[RatingPair] | Mapping[str, RatingPair]
Ranked = (
    IntoFrameT
    | tuple[ArrPairs, ArrVec]
    | tuple[Mapping[str, RatingPair], tuple[str]]
)
Columns = Literal['col', 'column', 'cols', 'columns', 1]
Rows = Literal['row', 'rows', 0]
Auto = Literal['auto', 'automatic']
XVec = Num[Any, 'x'] | Num[Any, '1 x']
DistVec = Num[Any, 'dist'] | Num[Any, 'dist 1']
DistMat = Num[Any, 'dist x']
PairLike = ArrPair | SeqPair

# %% Utils
ListOfDicts = list[dict[str, Any]]
DictOfLists = dict[str, list[Any]]

# %% Avoiding circular imports

# %% Pooling
# Old Type Aliases
# TODO(viamiraia): clean these up
TRealVec = NPVec | Sequence[Real]
SampRvs = Sequence[NPVec]
ContDist = FuzDist | rv_continuous_frozen
ContDists = Sequence[ContDist]
PoolNumF = Callable[[VecLike], NPVec]
PoolF = Callable[[VecLike], NPVec]
PoolNumFS = Callable[[], NPVec]
TPoolFS = Callable[[NPVec], NPVec]
