import os
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.activate = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, X):
        mid = self.activate(self.linear(X))
        out = self.output(mid)
        return out


# 测试代码
model = MyModel()
x = torch.randn((4, 3))
out = model(x)
list(model.named_parameters())

# 导出模型的权重，这个权重后续打包模型需要
OUT_DIR = r"D:\Downloads"
torch.save(model.state_dict(), os.path.join(OUT_DIR, 'myModel.pth'))

# 测试模型权重的加载
new_model = MyModel()
new_model.load_state_dict(torch.load(os.path.join(OUT_DIR, 'myModel.pth')))
list(new_model.named_parameters())

# 测试数据的处理
# data = [1.0, 2.0, 3.0]
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# data_tensor = torch.as_tensor(data, device=device)
# print(data_tensor.expand((1, -1)).shape)
