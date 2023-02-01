import torch


x = torch.ones((1, 2, 5, 10), dtype=torch.int8)

y = x.view(-1, 5)

# print(y)

a = torch.full((2, 3), 3)
# print(a)

y = torch.ones((2, 5, 10))

print(torch.add(x, y).shape)