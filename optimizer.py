from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                '''
                TODO: AdamW 구현을 완성하시오. 
                    위의 state 딕셔너리를 사용하여 상태를 읽고 저장하시오.
                    하이퍼파라미터는 `group` 딕셔너리에서 읽을 수 있다(생성자에 저장된 lr, betas, eps, weight_decay).

                        이 구현을 완성하기 위해서 해야할 일들:
                          1. 그래디언트의 1차 모멘트(첫 번째 모멘트)와 2차 모멘트(두 번째 모멘트)를 업데이트.
                          2. Bias correction을 적용(https://arxiv.org/abs/1412.6980 에 제공된 "efficient version" 사용; 프로젝트 설명의 pseudo-code에도 포함됨).
                          3. 파라미터(p.data)를 업데이트.
                          4. 그래디언트 기반의 메인 업데이트 후 weight decay 적용.

                        자세한 내용은 기본 프로젝트 안내문을 참조할 것.
                '''
                ### 완성시켜야 할 빈 코드 블록
                # 상태 초기화 (첫 호출 시)
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)  # m
                    state["exp_avg_sq"] = torch.zeros_like(p.data)  # v

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]

                state["step"] += 1
                step = state["step"]

                # 1. 업데이트 1차 모멘트 (m), 2차 모멘트 (v)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t = β1 * m_{t-1} + (1 - β1) * g_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t = β2 * v_{t-1} + (1 - β2) * g_t^2

                # 2. Bias correction (옵션)
                if correct_bias:
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    corrected_exp_avg = exp_avg / bias_correction1
                    corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                else:
                    corrected_exp_avg = exp_avg
                    corrected_exp_avg_sq = exp_avg_sq

                # 3. 파라미터 업데이트
                denom = corrected_exp_avg_sq.sqrt().add_(eps)
                step_size = alpha
                p.data.addcdiv_(corrected_exp_avg, denom, value=-step_size)

                # 4. weight decay 적용 (decoupled)
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-alpha * weight_decay)

        return loss
