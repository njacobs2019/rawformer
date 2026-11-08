from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from jaxtyping import Float
else:
    try:
        from jaxtyping import Float
    except ImportError:
        # jaxtyping is an optional dev/typecheck dependency. Without it,
        # Float[Tensor, "..."] degrades to plain Tensor.
        class Float:
            def __class_getitem__(cls, item: tuple[type, str]) -> type:
                return item[0]


RoPECache = tuple[Float[Tensor, "l rot_dim"], Float[Tensor, "l rot_dim"]]
Tokens = Float[Tensor, "b l d"]

__all__ = ["Float", "RoPECache", "Tokens"]
