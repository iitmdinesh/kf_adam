import torch
from torch.optim.optimizer import Optimizer


class KFAdam(Optimizer):
    r"""Implements the KFAdam optimization algorithm, which uses std deviation for the denominator rather than
    sqrt(energy) term used in conventional Adam. Why is this a good idea? If gradient stddev for a param is small, we
    should take larger steps as it means the gradient is consistent over time. The gradient and the standard deviation
    are estimated using a Kalman Filter instead of an EMA filter

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        beta: coefficient used for computing
            running averages of error variances (default: 0.95)
        step_size_limit: maximum step size (default: 1.0)
        eps: term added to the denominator to improve
            numerical stability (default: 1e-6)
    """

    def __init__(
            self,
            params,
            lr: float = 1e-3,
            beta: float = 0.95,
            step_size_limit: float = 1.0,
            eps: float = 1e-6,
    ):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if step_size_limit < 0.0:
            raise ValueError('Invalid step_size_limit value: {}'.format(step_size_limit))
        if not (0.0 < beta < 1.0):
            raise ValueError('Invalid beta value: {}'.format(beta))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))

        defaults = {
            'lr': lr,
            'beta': beta,
            'step_size_limit': step_size_limit,
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
            step_size_limit = group['step_size_limit']
            eps = group['eps']

            norm_sq = 0.0
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                norm_sq += grad.pow(2).sum()

            norm = torch.sqrt(norm_sq)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data / (eps + norm)
                if grad.is_sparse:
                    raise RuntimeError(
                        'KFAdam does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )

                state = self.state[p]
                if len(state) == 0:
                    sigma_init = (grad.pow(2).mean() + eps).sqrt()
                    state['iter'] = 1
                    state['estimate'] = torch.zeros_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state['estimate_error'] = 100 * sigma_init * torch.ones_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state['process_variance'] = sigma_init * torch.ones_like(
                        p.data, memory_format=torch.preserve_format
                    )
                    state['measurement_variance'] = sigma_init * torch.ones_like(
                        p.data, memory_format=torch.preserve_format
                    )
                prediction = state['estimate']
                measurement_variance = state['measurement_variance']
                process_variance = state['process_variance']
                prediction_error = state['estimate_error'] + process_variance
                kalman_gain = prediction_error / (eps + prediction_error + measurement_variance)
                innovation = grad - prediction
                estimate = prediction + kalman_gain * innovation
                estimate_error = (1.0 - kalman_gain).clamp(min=0.0) * prediction_error
                state['estimate_error'] = estimate_error
                state['estimate'] = estimate
                state['process_variance'] = process_variance * beta + (1.0 - beta) * innovation.pow(2)
                state['measurement_variance'] = measurement_variance * beta + (1.0 - beta) * (grad - estimate).pow(2)

                step = -lr * norm * estimate / ((estimate_error / (1.0 - beta ** state['iter'])).sqrt() + eps)
                step = step.clamp(-step_size_limit, step_size_limit)

                p.data.add_(step)
                state['iter'] += 1

        return loss