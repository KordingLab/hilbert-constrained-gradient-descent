# Hilbert Constrained Gradient Descent
Pytorch optimizers implementing Hilbert-Constrained Gradient Descent (HCGD) and Hilbert-Constrained Adam (HCADAM).
These optimizers modify their parent optimizers by penalizing the movement of the network in function space. We use the
$L^2$ notion of function space.

See Benjamin, Rolnick, Kording, ICLR 2019, https://openreview.net/forum?id=SkMwpiR9Y7

### Example usage

This optimizer is unusual in that it takes a 1st "test step" that it then corrects (in the direction
opposite the gradient of the change in function space, so that this change is minimized.) This behavior has the
following implications:
     - When calling .step(), we require that you supply a function that evaluates your network on validation data.
        This will be called after taking the "test step" to see how far in L2 space we've gone. See the example below.
     - There are hyperparameters that control the corrective step (`n_corrections`, `inner_lr`)


Notice the differences in the following training loop.
    - we loop over a validation dataloader `val_loader` as well as the normal training data
    - We feed a function called `validation_eval` to the `step()` call of the optimizer. This is necessary because
        the HC family of optimizers needs to internally re-calculate how far the network travels in function space
        after each internal loop.
    - We need to re-define `validation_eval` each loop to evaluate a particular batch of validation data.
```
# Code snippet... assuming the typical things like `model` is defined above

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
python sequential_mnist_example --learning-rate 0.01 --inner-learning-rate 0.08 --n-corrections 5 --opt hcadam
```
See the code for further documentation about this specific implementation.

### What is 'Hilbert-Constriction'?

We know that SGD implicitly constricts the movement traveled *in parameter space* every step, and that this is good for
optimization. We don't care about the parameters *per se*, but rather the input-output function itself. What we would
like to do is limit the change in function space each step.

There are many ways of defining a function space, and thus many ways of defining function distances. If you choose the
KL divergence between two output distributions as your measure of distance, it turns out that regularizing this notion of
change gives you the natural gradient. See the final section of our ICLR paper to see this made explicit.

In HC methods we use a notion of function distance that's somewhat easier to calculate. Given two functions, or networks,
the distance is simply the average difference between the outputs of those networks when given the same inputs. We call
this the $L^2$ function distance.

In math, the $L^2$ function distance between functions $f$ and $g$ is defined as:
$$\|f-g\|^2 \approx  \frac{1}{N}  \sum_{i=0}^N | f(x_i) -  g(x_i) |^2.$$

The HCGD and HCADAM optimizers both constrain how much this $L^2$ distance changes every optimization step. The procedure
is that we first take a "test step" using plain SGD or ADAM, compute how far we just traveled, and then step towards
the negative gradient of that distance with respect to the updated parameters, so that the distance decreases.

If computational resources allow, we can actually go a step further than just taking one corrective step. In analogy
with the natural gradient, we can work to converge towards a balance between the decrease in loss and the distance
traveled in $L^2$ space. If we approximate the change in loss linearly as the parameter change times the gradient,
we have a little mini-optimization problem each step, given by:
$$\Delta\theta'  = \argmin_{\Delta\theta} \bigg(J^T \Delta\theta+ \frac{\lambda}{N}  \sum_{i=0}^N | f_{\theta_t}(x_i) -  f_{\theta_{t}+\Delta\theta}(x_i) |^2 \bigg)$$
If you set `n_corrections` to be larger than 1, the HCGD and HCADAM optimizers will perform an inner loop of gradient
descent to converge towards the right $\Delta\theta$ each step. Performance usually improves if you use a few inner
steps, but be aware that the other hyperparameters may need to change along with the value of `n_corrections`.

### Fun with the $L^2$ distance

In our ICLR paper, we explore how typical networks behave in function space relative to parameter space. Here's a fun
image as a teaser.
![Function dists](Lipschitz-01.png)
Parameter distances is sometimes, but not always, representative of function distances.
Here we compare the two at three scales during the optimization of a CNN on CIFAR-10.
Left: Distances between the individual SGD updates. Middle: Distances between each epoch.
Right: Distances from initialization. On all three plots, note the changing relationship between function and
parameter distances throughout optimization. The network a CNN with four convolutional layers with batch normalization,
followed by two fully-connected layers, trained with SGD with learning rate = 0.1, momentum = 0.9, and weight decay = 1e-4.
Note that the $L^2$ distance is computed from the output after the softmax layer, meaning possible values range from 0 to 1.