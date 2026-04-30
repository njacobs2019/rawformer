from jaxtyping import Float
from torch import Tensor

RoPECache = tuple[Float[Tensor, "l rot_dim"], Float[Tensor, "l rot_dim"]]
Tokens = Float[Tensor, "b l d"]
