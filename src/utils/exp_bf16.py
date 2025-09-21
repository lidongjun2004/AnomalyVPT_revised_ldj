import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# 定义一个简单的 Transformer 模型
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = self.embedding(src)
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, model_dim)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, model_dim)
        output = self.fc_out(output)
        return output

# 初始化模型、损失函数和优化器
input_dim = 10
model_dim = 64
num_heads = 2
num_layers = 2
output_dim = 1

model = SimpleTransformer(input_dim, model_dim, num_heads, num_layers, output_dim).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始化GradScaler
scaler = GradScaler()

# 模拟输入数据
seq_len = 5
batch_size = 16
inputs = torch.randn(batch_size, seq_len, input_dim).cuda()
targets = torch.randn(batch_size, seq_len, output_dim).cuda()

# 训练循环
for epoch in range(10):
    optimizer.zero_grad()

    # 使用autocast启用fp16混合精度
    with autocast(dtype=torch.float16):
        outputs = model(inputs)
        # 检查中间张量的数据类型
        print(f"Outputs dtype: {outputs.dtype}")
        loss = criterion(outputs, targets)

    # 使用GradScaler进行梯度缩放
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')