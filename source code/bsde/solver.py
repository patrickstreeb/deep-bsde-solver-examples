"""
Deep BSDE solver (Han, Jentzen, E) for semilinear parabolic PDEs.
A PDE of the form

    0 = d_t u + mu(t,x) . grad u + 1/2 Tr(sigma sigma' D2 u) + f(t, x, u, sigma' grad u),
    u(T, x) = g(x),

is stored as a PDE object and rewritten as the forward-backward SDE

    X_{n+1} = X_n + mu(t_n, X_n) dt + sigma(t_n, X_n) dW_n,
    Y_{n+1} = Y_n - f(t_n, X_n, Y_n, Z_n) dt + Z_n . dW_n,      Y_T = g(X_T),

with the control Z = sigma' grad u.
"""

import numpy as np
import torch
from tqdm.auto import tqdm
from NN_model.networks import FeedForwardNet


def geometric_lr(lr, lr_end, it, num_iter):
    """Learning-rate schedule"""
    if not lr_end:
        return lr
    return lr * (lr_end / lr) ** (it / max(num_iter - 1, 1))


class DeepBSDESolver(torch.nn.Module):
    """
    Solves a PDE object with the deep BSDE method. Takes a PDE object defined per example.
    """

    def __init__(self, pde, net=FeedForwardNet, N=20, hidden=32, seed=0, y0_init=None,
                 z_init_scale=1.0):
        super().__init__()
        self.pde = pde
        self.N = int(N)
        self.dt = pde.T / self.N
        self.sqrt_dt = float(np.sqrt(self.dt))
        torch.manual_seed(seed)
        if y0_init is None:
            y0 = pde.terminal(pde.x0.unsqueeze(0))[0].clone()
        else:
            y0 = torch.full((pde.m,), float(y0_init))
        self.y0 = torch.nn.Parameter(y0)
        self.z0 = torch.nn.Parameter(torch.zeros(pde.m, pde.d))   # control at t = 0
        self.nets = torch.nn.ModuleList(
            [net(pde.d, pde.m * pde.d, hidden) for _ in range(self.N - 1)])   # one per step
        if z_init_scale != 1.0:
            for step_net in self.nets:
                last = [m for m in step_net.modules() if isinstance(m, torch.nn.Linear)][-1]
                with torch.no_grad():
                    last.weight.mul_(z_init_scale)
                    last.bias.mul_(z_init_scale)

    def _loss(self, batch):
        """Mean squared error in the terminal condition E[ sum (Y_T - g(X_T))^2 ]."""
        p = self.pde
        X = p.x0.expand(batch, p.d).clone()
        Y = self.y0.expand(batch, p.m).clone()
        Z = self.z0.expand(batch, p.m, p.d).clone()
        for n in range(self.N):
            t = n * self.dt
            dW = torch.randn(batch, p.d) * self.sqrt_dt
            Y = Y - p.f(t, X, Y, Z) * self.dt + (Z * dW.unsqueeze(1)).sum(dim=-1)
            mu = p.drift(t, X)
            dX = p.diffusion_dW(t, X, dW)
            if mu is not None:
                dX = dX + mu * self.dt
            X = X + dX
            if n < self.N - 1: # network input is scaled X
                Z = self.nets[n](p.normalize(X)).view(batch, p.m, p.d)
        return torch.mean(((Y - p.terminal(X)) ** 2).sum(dim=1))

    def solve(self, num_iter=2000, batch=256, lr=1e-2, lr_end=1e-3, net_lr_scale=1.0,
              progress=True):
        """
        Train the solver and return loss history
        """
        scales = (1.0, net_lr_scale)
        opt = torch.optim.Adam([
            {"params": [self.y0, self.z0], "lr": lr},
            {"params": self.nets.parameters(), "lr": lr * net_lr_scale},
        ])
        history = []
        self._reset_records()
        self.train()
        iterator = tqdm(range(num_iter), desc="training", unit=" iter") if progress \
            else range(num_iter)
        for it in iterator:
            for grp, scale in zip(opt.param_groups, scales):
                grp["lr"] = geometric_lr(lr, lr_end, it, num_iter) * scale
            opt.zero_grad()
            loss = self._loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            opt.step()
            history.append(float(loss.item()))
            self._record()
            if progress and it % 50 == 0:
                iterator.set_postfix(loss=f"{history[-1]:.4f}")
        return history

    def _reset_records(self):
        self.y0_history = []

    def _record(self):
        y0 = self.y0.detach().numpy().copy()
        self.y0_history.append(float(y0[0]) if y0.size == 1 else y0)

    @property
    def value(self):
        """helper"""
        y0 = self.y0.detach().numpy().copy()
        return float(y0[0]) if y0.size == 1 else y0
