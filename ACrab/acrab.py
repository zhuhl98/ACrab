# yapf: disable
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ACrab.util import compute_batched, DEFAULT_DEVICE, update_exponential_moving_average, normalized_sum


def l2_projection(constraint):
    @torch.no_grad()
    def fn(module):
        if hasattr(module, 'weight') and constraint>0:
            w = module.weight
            norm = torch.norm(w)
            w.mul_(torch.clip(constraint/norm, max=1))
    return fn

class ACrab(nn.Module):
    """ A-Crab: Actor-Critic Regularized by Average Bellman error """
    def __init__(self, *,
                 policy,
                 qf,
                 w,
                 target_qf=None,
                 optimizer,
                 discount=0.99,
                 Vmin=-float('inf'), # min value of Q (used in target backup)
                 Vmax=float('inf'), # max value of Q (used in target backup)
                 # Optimization parameters
                 policy_lr=5e-7,
                 qf_lr=5e-4,
                 w_lr=5e-4,
                 w_rs=0.5,
                 n_steps_w_opt=1,
                 target_update_tau=5e-3,
                 # ACrab parameters
                 reg='wave',
                 w_frac=0.5,
                 n_w=1,
                 w_a=1,
                 f_a=1,
                 C_infty=1,
                 # Entropy control
                 action_shape=None,  # shape of the action space
                 fixed_alpha=None,
                 target_entropy=None,
                 initial_log_alpha=0.,
                 # ATAC parameters
                 beta=1.0,  # the regularization coefficient in front of the Bellman error
                 norm_constraint=100,  # l2 norm constraint on the NN weight
                 # ATAC0 parameters
                 init_observations=None, # Provide it to use ATAC0 (None or np.ndarray)
                 buffer_batch_size=256,  # for ATAC0 (sampling batch_size of init_observations)
                 # Misc
                 debug=True,
                 ):

        #############################################################################################
        super().__init__()
        assert beta>=0 and norm_constraint>=0
        policy_lr = qf_lr if policy_lr is None or policy_lr < 0 else policy_lr # use shared lr if not provided.
        self._debug = debug  # log extra info

        # ATAC main parameter
        self.beta = beta # regularization constant on the Bellman surrogate

        # q update parameters
        self._discount = discount
        self._Vmin = Vmin  # lower bound on the target
        self._Vmax = Vmax  # upper bound on the target
        self._norm_constraint = norm_constraint  # l2 norm constraint on the qf's weight; if negative, it gives the weight decay coefficient.

        # networks
        self.policy = policy
        self._qf = qf
        self._target_qf = copy.deepcopy(self._qf).requires_grad_(False) if target_qf is None else target_qf
        self._w = w

        # optimization
        self._policy_lr = policy_lr
        self._qf_lr = qf_lr
        self._w_lr = w_lr
        self._alpha_lr = qf_lr # potentially a larger stepsize, for the most inner optimization.
        self._tau = target_update_tau
        
        # ACrab parameters
        self._n_steps_w_opt = n_steps_w_opt
        self.w_rs=w_rs
        self.reg=reg
        self.n_w=n_w
        self.w_frac=w_frac
        self.w_a = w_a
        self.f_a = f_a
        self.C_infty = C_infty

        self._optimizer = optimizer
        self._policy_optimizer = self._optimizer(self.policy.parameters(), lr=self._policy_lr) #  lr for warmstart
        self._qf_optimizer = self._optimizer(self._qf.parameters(), lr=self._qf_lr/self._n_steps_w_opt)
        self._w_optimizer = self._optimizer(self._w.parameters(), lr=self._w_lr/self._n_steps_w_opt)
        
        # control of policy entropy
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha
        if self._use_automatic_entropy_tuning:
            self._target_entropy = target_entropy if target_entropy else -np.prod(action_shape).item()
            self._log_alpha = torch.nn.Parameter(torch.tensor(initial_log_alpha))
            self._alpha_optimizer = optimizer([self._log_alpha], lr=self._alpha_lr)
        else:
            self._log_alpha = torch.tensor(self._fixed_alpha).log()

        # initial state pessimism (ATAC0)
        self._init_observations = torch.Tensor(init_observations) if init_observations is not None else init_observations  # if provided, it runs ATAC0
        self._buffer_batch_size = buffer_batch_size

    def update(self, log_func, upd_idx, observations, actions, next_observations, rewards, terminals, **kwargs):

        rewards = rewards.flatten()
        terminals = terminals.flatten().float()

        ##### Update Critic #####
        def compute_bellman_backup(q_pred_next):
            assert rewards.shape == q_pred_next.shape
            return (rewards + (1.-terminals)*self._discount*q_pred_next).clamp(min=self._Vmin, max=self._Vmax)
        
        # Pre-computation
        with torch.no_grad():  # regression target
            new_next_actions = self.policy(next_observations).sample()

        # qf_pred_both = self._qf.both(observations, actions)
        # qf_pred_next_both = self._qf.both(next_observations, new_next_actions)
        new_actions_dist = self.policy(observations)  # This will be used to compute the entropy
        new_actions = new_actions_dist.rsample() # These samples will be used for the actor update too, so they need to be traced.

        if self._init_observations is None:  # ACrab or ATAC
            pess_new_actions = new_actions.detach()
            pess_observations = observations
        else:  # initial state pessimism
            idx_ = np.random.choice(len(self._init_observations), self._buffer_batch_size)
            init_observations = self._init_observations[idx_]
            init_actions_dist = self.policy(init_observations)[0]
            pess_new_actions = init_actions_dist.rsample().detach()
            pess_observations = init_observations

        for _ in range(self._n_steps_w_opt):

            with torch.no_grad():
                target_q_values = self._target_qf(next_observations, new_next_actions)  # projection
                q_target = compute_bellman_backup(target_q_values.flatten())

            qf_pred_both, qf_pred_next_both, qf_new_actions_both \
                = compute_batched(self._qf.both, [observations, next_observations, pess_observations],
                                                [actions,      new_next_actions,  pess_new_actions])
            # construct loss function for w
            w_loss = 0
            w1, w2 = 1-self.w_rs, self.w_rs

            w_pred_tuple = self._w(observations, actions)

            for w_pred in w_pred_tuple: 
                for qfp, qfpn, qfna in zip(qf_pred_both, qf_pred_next_both, qf_new_actions_both):
                    # only update w
                    qfp, qfpn, qfna = qfp.detach(), qfpn.detach(), qfna.detach() 

                    assert qfp.shape == qfpn.shape == qfna.shape == q_target.shape == w_pred.shape
                    # compute abosolute average Bellman error
                    target_error = torch.abs(torch.mean(torch.mul(w_pred, torch.sub(qfp, q_target)))) - self.w_a *  torch.mean(torch.square(w_pred-1))/2 - torch.square(torch.mean(w_pred)-1)/2
                    q_backup = compute_bellman_backup(qfpn)  # compared with `q_target``, the gradient of `self._qf` is traced in `q_backup`.
                    residual_error = torch.abs(torch.mean(torch.mul(w_pred, torch.sub(qfp, q_backup)))) - self.w_a * torch.mean(torch.square(w_pred-1))/2 -  torch.square(torch.mean(w_pred)-1)/2
                    w_loss += w1*target_error + w2*residual_error

            w_loss = -w_loss

            # Update w
            self._w_optimizer.zero_grad()
            w_loss.backward()
            self._w_optimizer.step()
            self._w.apply(l2_projection(self._norm_constraint))

            # construct loss function for qf
            qf_loss = 0
            w1, w2 = 1-self.w_rs, self.w_rs

            # use two w networks and choose the max
            w_pred_1, w_pred_2 = self._w(observations, actions)
            w_pred_1 = w_pred_1.detach()
            w_pred_2 = w_pred_2.detach()

            for qfp, qfpn, qfna in zip(qf_pred_both, qf_pred_next_both, qf_new_actions_both):
                # Compute Bellman error
                assert qfp.shape == qfpn.shape == qfna.shape == q_target.shape

                w_rand = []
                if self.reg=='rand_w':
                    for _ in range(self.n_w):
                        rand_w = torch.abs(torch.randn_like(qfp, device=observations.device))
                        rand_w /= torch.mean(rand_w)
                        w_rand.append(rand_w)

                target_error_1 = torch.abs(torch.mean(torch.mul(w_pred_1, torch.sub(qfp, q_target))))
                target_error_2 = torch.abs(torch.mean(torch.mul(w_pred_2, torch.sub(qfp, q_target))))
                avg_error = (torch.max(target_error_1, target_error_2) + torch.abs(torch.mean(torch.sub(qfp, q_target)))) / 2
                target_error = self.w_frac * avg_error + (1-self.w_frac)*F.mse_loss(qfp, q_target)
                if self.reg=='abs':
                    target_error = self.w_frac * torch.mean(torch.abs(torch.sub(qfp, q_target))) + (1-self.w_frac)*F.mse_loss(qfp, q_target)
                if self.reg=='wapprox':
                    target_error = (2./self.beta) * self.w_frac * self.C_infty*torch.maximum(torch.mean(torch.max(qfp-q_target, torch.tensor([0.], device=qfp.device))), torch.mean(torch.max(q_target-qfp, torch.tensor([0.], device=qfp.device)))) + (1-self.w_frac)*F.mse_loss(qfp, q_target)
                if self.reg=='rand_w':
                    avg_bellman_error = torch.abs(torch.mean(torch.mul(w_rand[0], torch.sub(qfp, q_target)))) 
                    for idx in range(1,self.n_w):
                        avg_bellman_error += torch.abs(torch.mean(torch.mul(w_rand[idx], torch.sub(qfp, q_target))))
                    avg_bellman_error /= self.n_w
                    avg_bellman_error = (avg_bellman_error + torch.abs(torch.mean(torch.sub(qfp, q_target))))/2
                    target_error = self.w_frac *  avg_bellman_error + (1-self.w_frac)*F.mse_loss(qfp, q_target)
                

                q_backup = compute_bellman_backup(qfpn)  # compared with `q_target``, the gradient of `self._qf` is traced in `q_backup`.
                residual_error_1 = torch.abs(torch.mean(torch.mul(w_pred_1, torch.sub(qfp, q_backup))))
                residual_error_2 = torch.abs(torch.mean(torch.mul(w_pred_2, torch.sub(qfp, q_backup))))
                avg_error = (torch.max(residual_error_1, residual_error_2) + torch.abs(torch.mean(torch.sub(qfp, q_backup)))) / 2
                residual_error = self.w_frac * avg_error + (1-self.w_frac) * F.mse_loss(qfp, q_backup)
                if self.reg=='abs':
                    residual_error = self.w_frac*torch.mean(torch.abs(torch.sub(qfp, q_backup))) + (1-self.w_frac)*F.mse_loss(qfp, q_backup)
                if self.reg=='wapprox':
                    residual_error = (2./self.beta) * self.w_frac * self.C_infty*torch.maximum(torch.mean(torch.max(qfp-q_backup, torch.tensor([0.], device=qfp.device))), torch.mean(torch.max(q_backup-qfp, torch.tensor([0.], device=qfp.device)))) + (1-self.w_frac)*F.mse_loss(qfp, q_backup)
                if self.reg=='rand_w':
                    avg_bellman_error = torch.abs(torch.mean(torch.mul(w_rand[0], torch.sub(qfp, q_backup)))) 
                    for idx in range(1,self.n_w):
                        avg_bellman_error += torch.abs(torch.mean(torch.mul(w_rand[idx], torch.sub(qfp, q_backup))))
                    avg_bellman_error /= self.n_w
                    avg_bellman_error = (avg_bellman_error + torch.abs(torch.mean(torch.sub(qfp, q_backup))))/2
                    residual_error = self.w_frac * avg_bellman_error + (1-self.w_frac) * F.mse_loss(qfp, q_backup)
                

                qf_bellman_loss = w1*target_error+ w2*residual_error
                qf_bellman_loss += self.f_a * torch.mean(torch.square(qfp))/2
                # Compute pessimism term

                if self._init_observations is None:  # ACrab or ATAC
                    pess_loss = (qfna - qfp).mean()
                else:  # initial state pess. ATAC0
                    pess_loss = qfna.mean()
                ## Compute full q loss (qf_loss = pess_loss + beta * qf_bellman_loss)
                qf_loss += normalized_sum(pess_loss, qf_bellman_loss, self.beta)

            # Update q
            self._qf_optimizer.zero_grad()
            qf_loss.backward()
            self._qf_optimizer.step()
            self._qf.apply(l2_projection(self._norm_constraint))
            update_exponential_moving_average(self._target_qf, self._qf, self._tau)

        if upd_idx%10000 == 0:
                    log_func(f'qfna: {qfna}')
                    log_func(f'qfp: {qfp}')
                    log_func(f'qfna.mean: {qfna.mean()}')
                    log_func(f'qfp.mean: {qfp.mean()}')
                    log_func(f'w_pred_1: {w_pred_1}')
                    log_func(f'q_target: {q_target}')
                    log_func(f'q_backup: {q_backup}')

        ##### Update Actor #####
        # Compuate entropy
        log_pi_new_actions = new_actions_dist.log_prob(new_actions)
        policy_entropy = -log_pi_new_actions.mean()

        alpha_loss = 0
        if self._use_automatic_entropy_tuning:  # it comes first; seems to work also when put after policy update
            alpha_loss = self._log_alpha * (policy_entropy.detach() - self._target_entropy)  # entropy - target
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        # Compute performance difference lower bound (policy_loss = - lower_bound - alpha * policy_kl)
        alpha = self._log_alpha.exp().detach()
        self._qf.requires_grad_(False)
        lower_bound = self._qf.both(observations, new_actions)[-1].mean() # just use one network
        self._qf.requires_grad_(True)
        policy_loss = normalized_sum(-lower_bound, -policy_entropy, alpha)

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # Log
        log_info = dict(policy_loss=policy_loss.item(),
                        qf_loss=qf_loss.item(),
                        w_loss=w_loss.item(),
                        qf_bellman_loss=qf_bellman_loss.item(),
                        pess_loss=pess_loss.item(),
                        alpha_loss=alpha_loss.item(),
                        policy_entropy=policy_entropy.item(),
                        alpha=alpha.item(),
                        lower_bound=lower_bound.item())

        # For logging
        if self._debug:
            with torch.no_grad():
                debug_log_info = dict(
                        bellman_surrogate=residual_error.item(),
                        qf1_pred_mean=qf_pred_both[0].mean().item(),
                        qf2_pred_mean = qf_pred_both[1].mean().item(),
                        q_target_mean = q_target.mean().item(),
                        target_q_values_mean = target_q_values.mean().item(),
                        qf1_new_actions_mean = qf_new_actions_both[0].mean().item(),
                        qf2_new_actions_mean = qf_new_actions_both[1].mean().item(),
                        action_diff = torch.mean(torch.norm(actions - new_actions, dim=1)).item()
                        )
            log_info.update(debug_log_info)
        return log_info
