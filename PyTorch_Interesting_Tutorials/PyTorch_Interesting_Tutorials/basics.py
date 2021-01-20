import torch

if False:
    x = torch.tensor([5.5, 3])

    x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
    print(x)

    x = torch.randn_like(x, dtype=torch.float)    # override dtype!
    print(x)                                      # result has the same size

    print(x.size())

    y = torch.randn_like(x, dtype=torch.float)
    print(y)

    # adds x to y
    y.add_(x)
    print(y)

    x = torch.randn(4, 4)
    y = x.view(16)
    z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
    print(x.size(), y.size(), z.size())

    x = torch.randn(1)
    print(x)
    print(x.item())

    a = torch.ones(5)
    print(a)

    b = a.numpy()
    print(b)

    a.add_(1)
    print(a)
    print(b) # Numpy and torch.Tensor share the 
             # same memory location -> changing one will change other
    b += 1
    print(a)
    print(b)

    import numpy as np
    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out=a)
    print(a)
    print(b)

    if torch.cuda.is_available():
        x = torch.tensor([5.5, 3])
        device = torch.device("cuda")          # a CUDA device object
        y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
        x = x.to(device)                       # or just use strings ``.to("cuda")``
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!


    # Differentiation:

x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print(z, out)


    

out.backward()

print(x.grad)

x = x.add(5)

print(x.grad)