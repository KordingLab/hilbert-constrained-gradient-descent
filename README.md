# Hilbert Constrained Gradient Descent
Pytorch optimizers implementing Hilbert-Constrained Gradient Descent (HCGD) and Hilbert-Constrained Adam (HCADAM).
These optimizers modify their parent optimizers to perform gradient descent in function space, rather than parameter space. This is accomplished by penalizing the movement of the network in function space on each step while maintaining the decrease in the loss.

See Benjamin, Rolnick, Kording, ICLR 2019, https://openreview.net/pdf?id=SkMwpiR9Y7

An example implementation in a larger NMT package can be found here: https://github.com/aribenjamin/OpenNMT-py

## When to use

When will this approach work better? A heuristic we find helpful is this optimizer will perform better than SGD if Adam outperforms SGD. Think hard LSTM problems, but not CNNs or ResNets with batchnorm. **Pros:** Better generalization performance, especially for problems where the relation between functions and parameters is highly curved. **Cons:** ~2x training time, per step. Since it converges more quickly, this can cancel out in terms of wall-time.

We supply two different optimizers: HCadam and HCGD. Generally, HCadam works better than HCGD. See the paper for details on the difference between the two.

## Implementation

This optimizer requires the following change to your train loop:
- When calling .step(), we require that you supply a function that evaluates your network on validation data.
        This will be called after taking the "test step" to see how far in L2 space we've gone. See the example below.

This is because the optimizer takes a 1st "test step" that it then corrects (in the direction
opposite the gradient of the change in function space, so that this change is minimized). 

The default hyperparameter values are a good starting point. Try first searching over `fcn_change_limiter` within a factor of 10 on either side. Then, see if setting `n_corrections` to 2 or incrementally higher helps. This will get expensive, though, so consider whether it's worth your time. (Maybe you'd rather a bigger network, for example).

#### Example training loop
Notice the differences from a typical loop:
- We loop over a validation dataloader `val_loader` as well as the normal training data
- **We feed a function called `validation_eval` to the `step()` call of the optimizer**. This is necessary because
        the HC family of optimizers needs to internally re-calculate how far the network travels in function space
        after each internal loop.
- We need to re-define `validation_eval` each loop to evaluate a particular batch of validation data.
```
from hcgd import HCAdam, HCGD
...
optimizer = HCAdam(model.parameters(), lr=0.01,
                   fcn_change_limiter=1.0, inner_lr=0.05, n_corrections=5,  # HC params
                   betas=(0.9, 0.999), eps=1e-08, amsgrad=False)  # adam params


for current_batch, ((data, target), (val_data, _)) in enumerate(zip(training_data, val_loader)):
    # Do some normal things...
    if args.gpu:
        data, target, val_data = data.cuda(), target.cuda(), val_data.cuda()
    data, target, val_data = Variable(data), Variable(target), Variable(val_data)
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    # this is the key to using HC optimizers. We need to create a function that evaluates the model on the validation
    # batch, and then give this function to the step() call for the optimizer instance.
    def validation_eval():
        log_output = model(val_data)
        # We want to regularize the change in the function output, which should be the softmax.
        # Since pytorch classifers typically compute the log_softmax, we need to exponentiate the output.
        output =  torch.exp(log_output)
        return output

    # Finally, we feed this function we just defined to the optimizer as we step
    optimizer.step(validation_eval=validation_eval)

```

In the file `sequential_mnist_example.py`, we show an example of this in action. This can be run from the command line
right now with various options:
```
python sequential_mnist_example.py --learning-rate 0.01 --inner-learning-rate 0.08 --n-corrections 5 --opt hcadam
```
See the code for further documentation about this specific implementation.

### What is 'Hilbert-Constriction'?

Gradient descent takes the step with the largest decrease in loss for a fixed (small) distance in parameter space. But we don't care about the parameters *per se*, but rather the input-output function itself. What we would
like to do take the step with the largest decrease in loss for a fixed (small) distance in function space. That is, perform gradient descent in function space.

The way we actually do this is by pitting two factors against each other: on each step, we try to maximally decrease the loss while minimally moving in function space.

There are many ways of defining a function space, and thus many ways of defining function distances. If you choose the
KL divergence between two output distributions as your measure of distance, it turns out that regularizing this notion of
change gives you the natural gradient. See the final section of our ICLR paper to see this made explicit.

In HC methods we use a notion of function distance that's somewhat easier to calculate. Given two functions, or networks,
the distance is simply the average difference between the outputs of those networks when given the same inputs. We call
this the L^2 function distance.

In math, the L^2 function distance between functions f and g is defined as:

<img src="pics/eq1.png" alt="eq1" width="300"/>

The HCGD and HCADAM optimizers both constrain how much this L^2 distance changes every optimization step. The procedure
is that we first take a "test step" using plain SGD or ADAM, compute how far we just traveled, and then step towards
the negative gradient of that distance with respect to the updated parameters, so that the distance decreases.

If computational resources allow, we can actually go a step further than just taking one corrective step. In analogy
with the natural gradient, we can work to converge towards a balance between the decrease in loss and the distance
traveled in L^2 space. If we approximate the change in loss linearly as the parameter change times the gradient,
we have a little mini-optimization problem each step, given by:

<img src="pics/eq2.png" alt="eq1" width="400"/>

If you set `n_corrections` to be larger than 1, the HCGD and HCADAM optimizers will perform an inner loop of gradient
descent to converge towards the right ∆θ each step. Performance usually improves if you use a few inner
steps, but be aware that the other hyperparameters may need to change along with the value of `n_corrections`.

### Fun with the L^2 distance

In our ICLR paper, we explore how typical networks behave in function space relative to parameter space. Here's a fun
image as a teaser.
![Function dists](pics/Lipschitz-01.png)
Parameter distances is sometimes, but not always, representative of function distances.
Here we compare the two at three scales during the optimization of a CNN on CIFAR-10.
Left: Distances between the individual SGD updates. Middle: Distances between each epoch.
Right: Distances from initialization. On all three plots, note the changing relationship between function and
parameter distances throughout optimization. The network a CNN with four convolutional layers with batch normalization,
followed by two fully-connected layers, trained with SGD with learning rate = 0.1, momentum = 0.9, and weight decay = 1e-4.
Note that the L^2 distance is computed from the output after the softmax layer, meaning possible values range from 0 to 1.
