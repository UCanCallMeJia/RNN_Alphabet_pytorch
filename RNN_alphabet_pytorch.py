'''
A very simple but interesting
demo for Alphabet prediction.

Contact:jiazx@buaa.edu.cn
'''

import torch
from torch import nn,optim
import torchvision
import numpy as np

# define the length of training sequence
# Eg: if=5  "ABCDE" predict: "F"
SEQUENCE_LEN = 5
EPOCHS = 1000
LEARNING_RATE = 0.001

def Alpha_Dataset(seq_length = SEQUENCE_LEN):

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # map alpha to numbers
    alpha_to_num = dict((a, n) for n,a in enumerate(alphabet))
    # map numbers to alpha
    global num_to_alpha
    num_to_alpha = dict((n, a) for n,a in enumerate(alphabet))
    
    x = []
    y = []
    for index in range(len(alphabet)-seq_length):
        # get alpha sequence, like:'ABCDE','HIJKL'
        seq_in = alphabet[index:index+seq_length]
        # get corresponding predicted alpha, like:'F','M'
        seq_out = alphabet[index+seq_length]
        # convert to num sequence
        x.append([alpha_to_num[alpha] for alpha in seq_in])
        y.append(alpha_to_num[seq_out])
    x = np.reshape(np.array(x), (len(x),-1,1))/25.
    y = np.reshape(np.array(y), (len(y),-1))
    # show the shape of our toy dataset
    print('\n\n>>>>>>>>>>> Our Simple Dataset of Alphabet Prediction <<<<<<<<<<<\n',
            flush=True)
    print('\tX shape is: {}, Y shape is: {} '.format(x.shape, y.shape))
    print('\n>>>>>>>>>>>-------------------------------------------<<<<<<<<<<<\n\n')
    return x, y

class LSTM(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size = 1,
            hidden_size = 16,
            num_layers = 1,
            batch_first = True,
        )
        self.out = nn.Linear(
            in_features=16,
            out_features=26,
        )

    def forward(self, x):
        # None 代表初始化的隐藏状态全为0
        # h0 = torch.randn(2, 3, 20)
        rnn_out, (h,c) = self.rnn(x)
        # rnn_out 的size为: (batch, time_step, input)
        # 选取最后一个时间点作为输出
        out = self.out(rnn_out[:, -1, :])
        return out


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size = 1,
            hidden_size = 16,
            num_layers = 1,
            batch_first = True,
        )
        self.out = nn.Linear(
            in_features=16,
            out_features=26,
        )

    def forward(self, x):
        # None 代表初始化的隐藏状态全为0
        # h0 = torch.randn(2, 3, 20)
        rnn_out, h = self.rnn(x)
        # rnn_out 的size为: (batch, time_step, input)
        # 选取最后一个时间点作为输出
        out = self.out(rnn_out[:, -1, :])
        return out
        
if __name__ == "__main__":
    # load our Alpha dataset
    x, y = Alpha_Dataset()
    # shuffle dataset
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)

    # split training set (18,5,1) (18,1)
    x_train = torch.from_numpy(x)
    y_train = torch.from_numpy(y)

    # load a RNN model
    if torch.cuda.is_available():
        rnn = RNN().cuda()
    else:
        rnn = RNN()
    
    # define loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    opt = optim.Adam(rnn.parameters(), lr=LEARNING_RATE)

    step = 0
    for epoch in range(EPOCHS):
        for seq,label in zip(x_train,y_train):
            seq = seq.unsqueeze(0)
  
            # use GPU to train
            if torch.cuda.is_available():
                seq = seq.float().cuda()
                label = label.cuda()
            else:
                seq = torch.autograd.Variable(seq)
                label = torch.autograd.Variable(label)

            # 前向计算
            out = rnn(seq)
            # 计算损失
            loss = loss_func(out, label)   
            # 清空上一步优化器中的梯度
            opt.zero_grad()
            # 通过目标函数计算梯度
            loss.backward()
            # 优化器更新参数
            opt.step()

            if step%1000 == 0:
                print('step: {}, loss: {:.4}'
                    .format(step, loss.data.item()))
            step += 1
    
    # 测试模型
    rnn.eval()
    for seq,label in zip(x_train,y_train):
        seq = seq.unsqueeze(0)

        # use GPU to train
        if torch.cuda.is_available():
            seq = seq.float().cuda()
            label = label.cuda()
        else:
            seq = torch.autograd.Variable(seq)
            label = torch.autograd.Variable(label)

        # 前向计算
        out = rnn(seq)
        seq_num = seq.cpu().numpy()[0]*25
        # 转化为预测的字母
        pred_num = torch.argmax(out)
        pred_num = pred_num.cpu().numpy()
        pred_num = int(pred_num)
        
        seq_alpha = []
        for i in seq_num:
            global num_to_alpha
            seq_alpha.append(num_to_alpha[int(i)])
        print('Input:',seq_alpha,' Prediction:',num_to_alpha[pred_num])
