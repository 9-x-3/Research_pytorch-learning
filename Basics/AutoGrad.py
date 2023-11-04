import torch

#x = torch.randn(4, requires_grad=True)
#y = x + 2
#print(y)
#z = y*2
# z = z.mean()
#print(z)
# b = torch.tensor([0.2, 1.0,0.54,0.674] , dtype = torch.float32)
#z.backward() # or z.backward(b)
#print(x.grad)

# --both z.mean and backward with parameter creates a scalar vector to prevent error--

# methods to prevent history tracking of gradients
# 1 x.requires_grad_(False)
# 2 x.detach()
# with torch.no_grad(): some function

weights = torch.ones(6, requires_grad=True)

for epoch in range(4):
    model_output = (weights * 2).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()



