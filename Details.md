# KF-Adam: A Kalman Filter Based Optimizer for De-noising SGD

## Abstract

Stochastic gradients are inherently noisy due to minibatch sampling, and this noise can slow convergence and destabilize training.
This work describes *KF-Adam*, an optimizer that uses a Kalman-filter-style update to denoise stochastic gradient information before applying an Adam-like parameter update.
We summarize the underlying state--space view of noisy gradients, present the resulting filter equations, and outline an experimental protocol for evaluating the method.
The reference implementation is available at <https://github.com/iitmdinesh/kf_adam>.

## 1. Introduction

Training modern deep networks relies on variants of stochastic gradient descent (SGD).
Although minibatching makes optimization scalable, it introduces variance into the gradient estimate, which can lead to oscillatory updates and slower progress.
Adaptive optimizers such as Adam address some issues by scaling updates using running moment estimates, but the instantaneous gradient signal can remain noisy.

This work considers the gradient (or a transformed gradient statistic) as a latent, slowly varying quantity, observed through noisy minibatch measurements.
This perspective suggests applying a Kalman filter to produce a filtered (denoised) gradient estimate that can be fed into an adaptive optimizer.

## 2. Background

### 2.1 SGD and Adam

Let $\theta_t \in \mathbb{R}^d$ be parameters at step $t$.
Given a stochastic gradient observation $g_t = \nabla_\theta \ell(\theta_t; \mathcal{B}_t)$ from minibatch $\mathcal{B}_t$, SGD updates

$$
\theta_{t+1} = \theta_t - \eta\, g_t.
$$

Adam maintains exponential moving averages of first and second moments of gradients and uses bias correction to form an adaptive step.

### 2.2 Kalman filtering viewpoint

A (linear) Kalman filter assumes a latent state $x_t$ and noisy measurements $z_t$ with dynamics

$$
\begin{aligned}
x_t &= A x_{t-1} + w_t, & w_t \sim \mathcal{N}(0, Q), \\
z_t &= H x_t + v_t, & v_t \sim \mathcal{N}(0, R).
\end{aligned}
$$

In the optimizer setting, one can interpret the latent state as the "true" gradient statistic, and the measurement as the minibatch gradient.

## 3. KF-Adam Method

### 3.1 State--space model for gradient denoising

For clarity, consider a per-parameter (diagonal) filtering model.
Let $x_t \in \mathbb{R}^d$ be the latent denoised gradient at step $t$, and let $z_t \in \mathbb{R}^d$ be the observed minibatch gradient.
A simple random-walk model is

$$
\begin{aligned}
x_t &= x_{t-1} + w_t, & w_t \sim \mathcal{N}(0, Q), \\
z_t &= x_t + v_t, & v_t \sim \mathcal{N}(0, R),
\end{aligned}
$$

where $Q$ controls how quickly the latent gradient may change over time and $R$ models measurement (minibatch) noise.

With diagonal covariances, the Kalman predict--update reduces to elementwise operations:

$$
\begin{aligned}
\text{Predict:}\quad & \hat x^-_t = \hat x_{t-1}, \qquad P^-_t = P_{t-1} + Q, \\
\text{Gain:}\quad & K_t = \frac{P^-_t}{P^-_t + R}, \\
\text{Update:}\quad & \hat x_t = \hat x^-_t + K_t \odot (z_t - \hat x^-_t), \\
& P_t = (\mathbf{1} - K_t) \odot P^-_t.
\end{aligned}
$$

Here $\odot$ denotes elementwise multiplication.

### 3.2 Online noise estimation via running moments

In practice, the process/measurement noise levels ($Q$ and $R$) may be unknown and can vary over training.
A simple approach is to estimate them online using running (exponentially-weighted) moments of the filter residual (innovation) $\nu_t = z_t - \hat x^-_t$.
Following adaptive Kalman filtering ideas based on innovation covariance estimation [paper](https://article.nadiapub.com/IJCA/vol10_no10/6.pdf), one can update a diagonal estimate of the innovation covariance

$$
\widehat S_t = \beta\, \widehat S_{t-1} + (1-\beta)\, (\nu_t \odot \nu_t),
$$

and tie $R$ to $\widehat S_t$ (for example, $R_t = \widehat S_t$ or $R_t = \gamma\, \widehat S_t$ with a scalar $\gamma>0$).
This makes the Kalman gain $K_t$ respond to the observed gradient noise level, analogously to how Adam's second-moment estimate rescales updates.

### 3.3 Combining the filter with Adam

KF-Adam uses the filtered gradient estimate $\hat x_t$ (instead of $z_t$) as the input to an Adam-style moment update.
Concretely, replace $g_t$ with $\hat x_t$ wherever Adam would normally use the raw minibatch gradient.
This yields an adaptive update that (i) smooths noisy gradients and (ii) rescales steps using Adam's second-moment estimate.

### 3.4 Algorithm (high-level)

1. Compute stochastic gradient $z_t$ on minibatch.
2. Kalman filter update to obtain denoised gradient $\hat x_t$.
3. Apply Adam-style moment updates using $\hat x_t$.
4. Update parameters $\theta_{t+1}$.

## 4. Experiments (suggested protocol)

To evaluate KF-Adam, we recommend comparing against SGD, SGD+momentum, Adam, and AdamW on standard benchmarks.
Report (i) training loss vs. steps, (ii) validation accuracy vs. steps, and (iii) sensitivity to learning rate.

### 4.1 Implementation notes

The accompanying repository provides a reference implementation and usage examples.
When writing up results, include (a) filter hyperparameters (e.g., diagonal $Q$ and $R$ or their parameterization), (b) initialization of $P_0$, and (c) the precise way the filtered gradient is coupled to Adam's moments.

## 5. Conclusion

KF-Adam frames stochastic gradient noise as a measurement process and uses Kalman filtering to produce a denoised gradient signal for adaptive optimization.
Future work includes exploring structured (non-diagonal) covariance approximations and studying robustness under distribution shift.
