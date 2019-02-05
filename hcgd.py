from torch.optim import Optimizer
import torch
required = object()
import math


class HCGD(Optimizer):
    """This optimizer modifies SGD by adding the gradient of an auxiliary cost to decrease the step size in
    L2 function space.

    From Benjamin, Rolnick, Kording, ICLR 2019, https://openreview.net/forum?id=SkMwpiR9Y7

    This optimizer is unusual in that it takes a 1st "test step" that it then corrects (in the direction
    opposite the gradient of the change in function space, so that this change is minimized.) This behavior has the
    following implications:
         - when calling .step(), we require that you supply a function that evaluates your network. This will be called
            after taking the "test step" to see how far in L2 space we've gone. See the example below.
         - there are hyperparameters that control the corrective step (`n_corrections`, `inner_lr`)
         - We call 'zero_grad()` internally, so don't use this if you want gradient information to stay stored
            after you call `step()`


    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate
        fcn_change_limiter: relative strength of the functional regularization compared to the overall cost
        n_corrections: number of iterations in the inner loop
        inner_lr: learning rate in the inner loop. Usually 1/10 of the main learning rate works well.
        clip_correction_grad: what to clip the grads on the correction to (good if you're clipping other grads too).
                                Default of 0 means no clipping.

        Inherited from Adam:
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        ams_grad: See Adam docs;

    Example:
        >>>

    """

    def __init__(self, params, lr=required, momentum=0, fcn_change_limiter=0.5, inner_lr=0.01,
                 n_corrections=1, clip_correction_grad=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, fcn_change_limiter=fcn_change_limiter,
                        inner_lr=inner_lr)

        super(HCGD, self).__init__(params, defaults)
        self.n_corrections = n_corrections
        self.clip_correction_grad = clip_correction_grad

    def __setstate__(self, state):
        super(HCGD, self).__setstate__(state)

    def step(self, validation_eval, closure=None, orig_val_output=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that re-evaluates the model, runs `backward`,
                and returns the loss.

            validation_eval (callable): a function that returns `model(val_data)`,
                            where `val_data` is a batch different from the current training batch.

            orig_val_output: The outputs of validation_eval, evaluated before the network steps.
                                If not supplied, we automatically evaulate it anyways.
                                This option is here because there are situations in which one evaluates
                                the network on validation data anyways, in which case it would be silly to
                                re-evaluate it here
        """

        loss = None
        if closure is not None:
            loss = closure()

        if orig_val_output is None:
            orig_val_output = validation_eval().detach()

        # store Jacobian (only needed if n_corrections is >1)
        if self.n_corrections > 1:
            jacobian = [[None if p.grad is None else p.grad.clone() for p in group['params']]
                        for group in self.param_groups]
        else:
            # make empty (just a placeholder that we'll iterate through later)
            jacobian = [[None for p in group['params']] for group in self.param_groups]

        # take a proposed step.
        # if using momentum, the proposed step includes momentum
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if d_p.is_sparse:
                    raise RuntimeError('HCGD does not support sparse gradients.')

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                v = d_p.mul(lr)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'velocity' not in param_state:
                        v = param_state['velocity'] = torch.zeros_like(p.data)
                        v.mul_(momentum).add_(lr, d_p)
                    else:
                        v = param_state['velocity']
                        v.mul_(momentum).add_(lr, d_p)
                p.data.add_(-1, v)

        for i in range(self.n_corrections):

            self.zero_grad()

            # get change in output on validation data now that we've updated the data
            prop_val_output = validation_eval()

            output_diff = torch.norm(prop_val_output - orig_val_output, 2, 1).mean()

            # get the derivative and accumulate into p.grad
            output_diff.backward()

            # clip the grads on the correction (good if you're clipping other grads too)
            if self.clip_correction_grad > 0:
                # clip groups separately
                for group in self.param_groups:
                    torch.nn.utils.clip_grad_norm(group['params'], self.clip_correction_grad)

            # step towards this value
            for group, group_jacobian in zip(self.param_groups, jacobian):
                weight_decay = group['weight_decay']
                fcn_change_limiter = group['fcn_change_limiter']
                inner_lr = group['inner_lr']
                momentum = group['momentum']

                for p, j in zip(group['params'], group_jacobian):
                    if (p.grad is None) and (j is None):
                        continue

                    d_p = p.grad.data

                    # add original jacobian and d_p with d_p scaled
                    if i == 0:
                        jac = d_p.mul(fcn_change_limiter)
                    else:
                        jac = j.data
                        jac.add_(fcn_change_limiter, d_p)

                    # update the network with the correction
                    p.data.add_(-inner_lr, jac)

                    # now update the momentum buffer
                    # (we want it later. it's not used for the correction)
                    if momentum != 0:
                        param_state = self.state[p]
                        v = param_state['velocity']
                        # add the correction
                        v.add_(inner_lr, jac)

        return loss


class HCAdam(Optimizer):
    """This optimizer modifies Adam by adding the gradient of an auxiliary cost to decrease the step size in
    L2 function space.

    From Benjamin, Rolnick, Kording, ICLR 2019, https://openreview.net/forum?id=SkMwpiR9Y7

    This optimizer is unusual in that it takes a 1st "test step" that it then corrects (in the direction
    opposite the gradient of the change in function space, so that this change is minimized.) This behavior has the
    following implications:
         - when calling .step(), we require that you supply a function that evaluates your network. This will be called
            after taking the "test step" to see how far in L2 space we've gone. See the example below.
         - there are hyperparameters that control the corrective step (`n_corrections`, `inner_lr`)
         - We call 'zero_grad()` internally, so don't use this if you want gradient information to stay stored
            after you call `step()`

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        fcn_change_limiter: relative strength of the functional regularization compared to the overall cost
        n_corrections: number of iterations in the inner loop
        inner_lr: learning rate in the inner loop. Usually 1/10 of the main learning rate works well.
        clip_correction_grad: what to clip the grads on the correction to (good if you're clipping other grads too).
                                Default of 0 means no clipping.

        Inherited from Adam:
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        ams_grad: See Adam docs;


    Example:
        >>>

    __ http:



    """


    def __init__(self, params, lr=required, fcn_change_limiter=0.5, inner_lr=0.01,
                 n_corrections=1, clip_correction_grad=0, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):

        defaults = dict(lr=lr, fcn_change_limiter=fcn_change_limiter,
                        inner_lr=inner_lr,
                        betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)

        super(HCAdam, self).__init__(params, defaults)
        self.n_corrections = n_corrections
        self.clip_correction_grad = clip_correction_grad


    def __setstate__(self, state):
        super(HCAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)


    def step(self, validation_eval, closure=None, orig_val_output=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model, runs `backward`,
                and returns the loss.

            validation_eval (callable): a function that returns `model(val_data)`,
                            where `val_data` is a batch different from the current training batch.

            orig_val_output: The outputs of validation_eval, evaluated before the network steps.
                                If not supplied, we automatically evaulate it anyways.
                                This option is here because there are situations in which one evaluates
                                the network on validation data anyways, in which case it would be silly to
                                re-evaluate it here
        """

        loss = None
        if closure is not None:
            loss = closure()

        if orig_val_output is None:
            orig_val_output = validation_eval().detach()


        # store Jacobian (only needed if n_corrections is >1)
        if self.n_corrections > 1:
            jacobian = [[None if p.grad is None else p.grad.clone() for p in group['params']]
                        for group in self.param_groups]
        else:
            # make empty (just a placeholder that we'll iterate through later)
            jacobian = [[None for p in group['params']]
                        for group in self.param_groups]


        # In this loop we just take a step of Adam
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                amsgrad = group['amsgrad']
                if grad.is_sparse:
                    raise RuntimeError('HCAdam does not support sparse gradients.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                # get the relevant state params.
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                ### Take Adam step
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)


        # Now, once we've taken a "test step", we correct this back towards the original location (with the
        # distance measure being L2 distance)
        for i in range(self.n_corrections):

            self.zero_grad()

            # get change in output on validation data now that we've updated the data
            prop_val_output = validation_eval()

            diff = prop_val_output - orig_val_output
            distance_per_example = torch.norm(diff, 2, 1)
            L2_mean = distance_per_example.mean()

            # get the derivative and accumulate into p.grad
            L2_mean.backward()

            # clip the grads on the correction (good if you're clipping other grads too)
            if self.clip_correction_grad > 0:
                # clip groups separately
                for group in self.param_groups:
                    torch.nn.utils.clip_grad_norm(group['params'], self.clip_correction_grad)

            # step towards this value
            for group, group_jacobian in zip(self.param_groups, jacobian):
                fcn_change_limiter = group['fcn_change_limiter']
                inner_lr = group['inner_lr']

                for p, j in zip(group['params'], group_jacobian):
                    if (p.grad is None) and (j is None):
                        continue
                    amsgrad = group['amsgrad']
                    L2_grad = p.grad.data

                    if i == 0:
                        jac = L2_grad.mul(fcn_change_limiter)
                    else:
                        # this is tricky: for n_iterations>1, we don't continue to converge towards the starting point;
                        # instead, we attempt to converge to the point such that
                        #               ``  jacobian = -fcn_change_limiter * L2_grad    ``
                        jac = j.data
                        jac.add_(fcn_change_limiter, L2_grad)

                    # update the network with the correction
                    p.data.add_(-inner_lr, jac)

                    # update the Adam state params
                    # (we want it later. it's not used for the correction)

                    state = self.state[p]
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                    beta1, beta2 = group['betas']
                    # Add the correction into the first and second running moments
                    exp_avg.add_(1 - beta1, L2_grad)
                    exp_avg_sq.addcmul_(1 - beta2, L2_grad, L2_grad)

                    if amsgrad:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

        return loss