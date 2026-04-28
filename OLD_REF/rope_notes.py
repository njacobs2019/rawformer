import torch


def rotate_every_two(x):
    # x: (..., dim)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    print("x1")
    print(x1)

    print("x2")
    print(x2)

    print("stack")
    print(torch.stack((-x2, x1), dim=-1))

    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


#####

x = torch.arange(1, 11).reshape(1, -1)
print(x.shape)
print(x)

#####

print(rotate_every_two(x))
