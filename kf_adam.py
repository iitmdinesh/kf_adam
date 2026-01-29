import torch
from torch.optim.optimizer import Optimizer


class KFAdam(Optimizer):
    r"""Implements the KFAdam optimization algorithm. The gradient and the standard deviation
    are estimated using a Kalman Filter instead of an EMA filter.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        beta: coefficient used for computing
            running averages of error variances (default: 0.95)
        eps: term added to the denominator to improve
            numerical stability (default: 1e-12)
    """

    def __init__(
            self,
            params,
            lr: float = 1e-3,
            beta: float = 0.95,
            eps: float = 1e-12,
    ):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not (0.0 < beta < 1.0):
            raise ValueError('Invalid beta value: {}'.format(beta))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))

        defaults = {
            'lr': lr,
            'beta': beta,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            lr = group['lr']
            beta = group['beta']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                where = grad != 0
                if grad.is_sparse:
                    raise RuntimeError(
                        'KFAdam does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                state = self.state[p]
                if len(state) == 0:
                    sigma_sq_init = grad[where].pow(2).mean()
                    state['iter'] = 1
                    state['estimate'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state['obs_prev'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state['obs_one_before_prev'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state['estimate_error'] = sigma_sq_init
                    state['obs_prev_var'] = sigma_sq_init
                    state['obs_one_before_prev_var'] = sigma_sq_init
                state['obs_prev_var'] = \
                    state['obs_prev_var'] * beta + (1 - beta) * (grad - state['obs_prev'])[where].pow(2).mean()
                state['obs_one_before_prev_var'] = state['obs_one_before_prev_var'] * beta + \
                    (1 - beta) * (grad - state['obs_one_before_prev'])[where].pow(2).mean()
                # Reference: http://article.nadiapub.com/IJCA/vol10_no10/6.pdf
                measurement_variance = (state['obs_one_before_prev_var'] - 0.5 * state['obs_prev_var'])).clamp(min=eps)
                process_variance = (state['obs_prev_var'] - 2 * measurement_variance).clamp(min=eps)
                prediction = state['estimate']
                prediction_error = state['estimate_error'] + process_variance
                kalman_gain = (prediction_error / (eps + prediction_error + measurement_variance)).clamp(min=0.0, max=1.0)
                innovation = grad - prediction
                estimate = prediction + kalman_gain * innovation
                estimate_error = (1.0 - kalman_gain) * prediction_error
                state['estimate_error'] = estimate_error
                state['estimate'] = torch.where(where, estimate, state['estimate'])

                step = -lr * estimate / (torch.sqrt(estimate_error + estimate.pow(2)) + eps)
                step = torch.where(where, step, torch.zeros_like(step))

                p.data.add_(step)
                state['iter'] += 1
                state['obs_one_before_prev'] = torch.where(where, state['obs_prev'], state['obs_one_before_prev'])
                state['obs_prev'] = torch.where(where, grad, state['obs_prev'])

        return loss
