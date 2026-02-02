"""
优化器和学习率调度器
"""

import numpy as np


class ScheduledOptim:
    """
    带Warmup的学习率调度器
    
    实现了Transformer论文中的学习率策略:
        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    
    特点:
    - 前warmup_steps步内，学习率线性增长
    - 之后，学习率按step^(-0.5)衰减
    
    使用示例:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98))
        scheduler = ScheduledOptim(optimizer, d_model=512, n_warmup_steps=4000)
        
        # 训练循环中
        for batch in dataloader:
            scheduler.zero_grad()
            loss = compute_loss(batch)
            loss.backward()
            scheduler.step_and_update_lr()  # 同时更新参数和学习率
    """

    def __init__(self, optimizer, d_model, n_warmup_steps, lr_mul=1.0):
        """
        Args:
            optimizer: PyTorch优化器实例 (如Adam)
            d_model: 模型维度 (用于计算学习率缩放)
            n_warmup_steps: Warmup步数
            lr_mul: 学习率乘数，用于缩放最终学习率（默认1.0）
                   建议值：对于d_model=256，使用10-20
        """
        self._optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        self.lr_mul = lr_mul  # 新增：学习率缩放因子

    def step_and_update_lr(self):
        """更新参数并调整学习率"""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """清空梯度"""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        """
        计算当前步的学习率缩放因子
        
        Returns:
            float: 学习率缩放因子
        """
        d_model = self.d_model
        n_steps = self.n_steps
        n_warmup_steps = self.n_warmup_steps

        # Transformer论文中的学习率调度公式
        base_lr = (d_model ** -0.5) * min(n_steps ** -0.5, n_steps * n_warmup_steps ** -1.5)
        
        # 应用学习率乘数（修复小模型学习率过低的问题）
        return base_lr * self.lr_mul

    def _update_learning_rate(self):
        """根据当前步数更新学习率"""
        self.n_steps += 1
        lr = self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_current_lr(self):
        """获取当前学习率"""
        return self._optimizer.param_groups[0]['lr']
    
    def state_dict(self):
        """保存调度器状态"""
        return {
            'n_steps': self.n_steps,
            'd_model': self.d_model,
            'n_warmup_steps': self.n_warmup_steps,
            'lr_mul': self.lr_mul,
            'optimizer': self._optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.n_steps = state_dict['n_steps']
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.lr_mul = state_dict.get('lr_mul', 1.0)  # 向后兼容
        self._optimizer.load_state_dict(state_dict['optimizer'])


# ============================================================================
# 可视化学习率曲线（可选工具）
# ============================================================================

def plot_lr_schedule(d_model=512, n_warmup_steps=4000, max_steps=50000):
    """
    可视化学习率调度曲线
    
    Args:
        d_model: 模型维度
        n_warmup_steps: Warmup步数
        max_steps: 最大步数
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要安装matplotlib: pip install matplotlib")
        return
    
    steps = np.arange(1, max_steps + 1)
    lrs = []
    
    for step in steps:
        lr = (d_model ** -0.5) * min(step ** -0.5, step * n_warmup_steps ** -1.5)
        lrs.append(lr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs)
    plt.axvline(x=n_warmup_steps, color='r', linestyle='--', label=f'Warmup End (step={n_warmup_steps})')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title(f'Transformer Learning Rate Schedule (d_model={d_model}, warmup={n_warmup_steps})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lr_schedule.png', dpi=150)
    plt.show()
    print("学习率曲线已保存至 lr_schedule.png")


if __name__ == "__main__":
    import torch
    
    # 示例: 创建带warmup的优化器
    dummy_param = torch.nn.Parameter(torch.randn(10, 10))
    optimizer = torch.optim.Adam([dummy_param], lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = ScheduledOptim(optimizer, d_model=512, n_warmup_steps=4000)
    
    print("="*60)
    print("学习率调度示例")
    print("="*60)
    
    # 显示前10步和warmup结束附近的学习率
    test_steps = [1, 10, 100, 1000, 4000, 8000, 16000]
    
    for step in test_steps:
        scheduler.n_steps = step
        lr = scheduler._get_lr_scale()
        print(f"Step {step:6d}: lr = {lr:.6f}")
    
    print("\n提示: 运行 plot_lr_schedule() 可以可视化完整曲线")
    
    # 取消注释以生成学习率曲线图
    # plot_lr_schedule()
