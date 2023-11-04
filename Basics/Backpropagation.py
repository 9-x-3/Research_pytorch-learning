import torch


w = torch.tensor(1.0, requires_grad=True)

# defining hyperparameters

learning_rate = 0.01
num_epochs = 10

for epoch in range(num_epochs):
    x = torch.tensor(1.0)
    y = torch.tensor(2.0)

# first step do Forward pass and compute the loss
    y_hat = w * x
    loss = (y_hat - y)**2
    print(loss)

# backward pass
    loss.backward()


# update weights
# next forward pass and backward pass

   # update the weight with negative gradient
    with torch.no_grad():
        w-= learning_rate * w.grad

        # remove the gradient to prepare for next iteration
        w.grad.zero_()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Updated weight: {w.item()}')
