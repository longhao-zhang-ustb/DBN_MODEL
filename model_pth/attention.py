import torch
import torch.nn as nn

class SelfAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super(SelfAttentionLayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert self.hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, q, k, v, mask=None):
        '''
        :param q:   shape [batch_size, seq_length, hid_dim]
        :param k:   shape [batch_size, seq_length, hid_dim]
        :param v:   shape [batch_size, seq_length, hid_dim]
        :param mask:
        :return:
        '''
        batch_size = q.shape[0]

        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # Q,K,V shape [batch_size, n_heads, seq_length, hid_dim // n_heads]

        Q = Q.contiguous().view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.contiguous().view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.contiguous().view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # energy [batch_size, n_heads, seq_length, seq_length]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        # attention [batch_size, n_heads, seq_length, seq_length]
        attention = self.dropout(torch.softmax(energy, dim=-1))
        # x [batch_size, n_heads, seq_length, hid_dim // n_heads]
        x = torch.matmul(attention, V)

        x = x.contiguous().permute(0, 2, 1, 3)
        # x [batch_size, seq_length, hid_dim]
        x = x.contiguous().view(batch_size, -1, self.n_heads * (self.hid_dim // self.n_heads))

        x = self.fc(x)

        if mask is not None:
            mask = mask.squeeze(1).squeeze(1)
            mask = mask.unsqueeze(2).repeat(1, 1, self.hid_dim).float()
            x = x * mask
        # [batch_size, seq_length, hid_dim]
        return x

# https://blog.51cto.com/u_16099350/7882738
# if __name__ == '__main__':
#     input=torch.randn(50,49,512)
#     sa = SelfAttentionLayer(hid_dim=512, n_heads=4, dropout=.3, device="cpu")
#     output=sa(input,input,input)
#     print(output.shape)