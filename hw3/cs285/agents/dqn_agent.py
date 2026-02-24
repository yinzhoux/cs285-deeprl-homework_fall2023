from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu

def assert_tensor_shape(shape, tensor):
    assert tensor.shape == shape, f'{tensor.shape}'
def assert_tensor_shape_eq(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape, f'{tensor1.shape} vs. {tensor2.shape}'

class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        logits: torch.Tensor = self.critic(observation) # action_cnt-dim
        greedy_action = torch.argmax(logits, dim=-1, keepdim=True)
        p_exploration = epsilon / (logits.shape[-1] - 1)
        p_greedy = 1 - epsilon
        prob = torch.full(logits.shape, p_exploration, device=logits.device)
        prob.scatter_(1, greedy_action, p_greedy)
        dist = torch.distributions.Categorical(probs=prob)
        action = dist.sample()

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(student): compute target values
            next_qa_values_target: torch.Tensor = self.target_critic(next_obs) # [batch size, action nums]
            assert_tensor_shape((batch_size, self.num_actions), next_qa_values_target)

            if self.use_double_q: # select action from critic
                next_qa_values = self.critic(next_obs)
                _, next_action = next_qa_values.max(dim=-1)
            else: # select action from target critic
                _, next_action = next_qa_values_target.max(dim=-1) # [batch size,]
            
            assert_tensor_shape((batch_size,), next_action)
            next_action.unsqueeze_(dim=-1) # [batch_size, 1]

            next_q_values = torch.gather(next_qa_values_target, 1, next_action) # [batch size, 1]
            next_q_values.squeeze_(dim=-1) # [batch size, ]
            assert_tensor_shape_eq(reward, next_q_values)
            target_values = reward + self.discount * next_q_values * (1 - done.float()) # [batch size, ]

        # TODO(student): train the critic with the target values
        qa_values = self.critic(obs) # [batch size, action nums]
        q_values = torch.gather(qa_values, 1, action.unsqueeze(dim=-1)).squeeze_(dim=-1) # Compute from the data actions; see torch.gather
        assert_tensor_shape_eq(q_values, target_values)
        loss = self.critic_loss(q_values, target_values) # both [batch size, ]

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): update the critic, and the target if needed
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)
        if step % self.target_update_period == 0:
            self.update_target_critic()
        return critic_stats
