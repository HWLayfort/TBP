import torch
import torch.nn as nn

# -------------------------------
# 1) DeepONet
# -------------------------------
class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim,
                 hidden_dim=128, branch_layers=3, trunk_layers=3, out_dim=9,
                 activation=nn.Tanh):
        super().__init__()
        layers = []
        in_dim = branch_input_dim
        for _ in range(branch_layers):
            layers += [nn.Linear(in_dim, hidden_dim), activation()]
            in_dim = hidden_dim
        self.branch_net = nn.Sequential(*layers)

        layers = []
        in_dim = trunk_input_dim
        for _ in range(trunk_layers):
            layers += [nn.Linear(in_dim, hidden_dim), activation()]
            in_dim = hidden_dim
        self.trunk_net = nn.Sequential(*layers)

        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, branch_input, trunk_input):
        b = self.branch_net(branch_input)
        t = self.trunk_net(trunk_input)
        return self.fc(b * t)


# ---------------------------------------
# 2) SpectralConv1d & FNO1d
# ---------------------------------------
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.weights = nn.Parameter(
            (1/(in_channels*out_channels)) *
            torch.randn(in_channels, out_channels, modes1, dtype=torch.cfloat)
        )
        self.modes1 = modes1

    def forward(self, x):
        batch, _, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(batch, self.weights.shape[1], x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = torch.einsum(
            "bci,cio->boi", x_ft[:, :, :self.modes1], self.weights
        )
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, in_channels, out_channels, width, modes1, depth):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, 1)
        self.spec_blocks = nn.ModuleList(
            [SpectralConv1d(width, width, modes1) for _ in range(depth)]
        )
        self.pw_blocks = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(depth)]
        )
        self.proj1 = nn.Conv1d(width, width//2, 1)
        self.proj2 = nn.Conv1d(width//2, out_channels, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.lift(x)
        for spec, pw in zip(self.spec_blocks, self.pw_blocks):
            x = self.act(x + spec(x) + pw(x))
        return self.proj2(self.act(self.proj1(x)))


# ---------------------------------------
# 3) PINN & rPINN
# ---------------------------------------
class PINN(nn.Module):
    def __init__(self, hidden_dim=64, num_hidden_layers=4,
                 input_dim=1, output_dim=9):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Tanh()

    def forward(self, t):
        x = self.act(self.input_layer(t))
        for h in self.hidden_layers:
            x = self.act(h(x))
        return self.output_layer(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.Tanh()

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(x + out)


class rPINN(nn.Module):
    def __init__(self, input_dim=1, output_dim=9,
                 hidden_dim=64, num_res_blocks=4):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Tanh()

    def forward(self, t):
        x = self.act(self.input_layer(t))
        for blk in self.res_blocks:
            x = blk(x)
        return self.output_layer(x)


# ---------------------------------------
# 4) PINN/rPINN 전용 helper
# ---------------------------------------
def compute_derivatives(model, t):
    t = t.clone().requires_grad_(True)
    r = model(t)
    v = torch.zeros_like(r)
    for i in range(r.shape[1]):
        v[:, i] = torch.autograd.grad(r[:, i].sum(), t, create_graph=True)[0].squeeze(-1)
    a = torch.zeros_like(r)
    for i in range(r.shape[1]):
        a[:, i] = torch.autograd.grad(v[:, i].sum(), t, create_graph=True)[0].squeeze(-1)
    return r, v, a


def physics_residual(r, a, masses, G=1.0):
    r = r.view(-1, 3, 3)
    a = a.view(-1, 3, 3)
    diff = r.unsqueeze(2) - r.unsqueeze(1)
    norm = torch.norm(diff, dim=-1, keepdim=True) + 1e-6
    m = torch.tensor(masses, device=r.device).view(1,1,3,1)
    a_grav = G * m * diff / norm.pow(3)
    mask = 1 - torch.eye(3, device=r.device).view(1,3,3,1)
    a_grav = (a_grav * mask).sum(dim=2)
    return (a - a_grav).view(-1, 9)
