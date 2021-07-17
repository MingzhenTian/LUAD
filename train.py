# 训练模型部分
import torch
from model import resnet18
from datamanager import obtain_loader
import torch.optim as optim
import os


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train_loop(model, data_loader, optimizer,epoch):
    avg_loss = 0.0
    correct = 0.0
    total = 0.0

    model.train()
    train_result = open('train_result150to200.txt', "a")
    for i, (x, label) in enumerate(data_loader):
        x, label = x.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, label)
        loss.backward()
        _, predicted = torch.max(out.data, 1)
        correct += predicted.eq(label.data).cpu().sum()
        optimizer.step()
        avg_loss = (avg_loss * i + loss.data.item()) / float(i + 1)
        total += label.size(0)
    print('Train Epoch: {} avg_loss:{} accuracy:{}'.format(epoch, avg_loss, 100. * correct / total))
    train_result.write('Train Epoch: {} avg_loss:{} accuracy:{}'.format(epoch, avg_loss, 100. * correct / total))
    train_result.write('\n')
    train_result.close()


def test_loop(model, data_loader,epoch):
    correct = 0
    total=0
    model.eval()
    test_result=open('test_result150to200.txt', 'a')
    with torch.no_grad():
        for i,(x, label) in enumerate(data_loader):
            x, label = x.to(device), label.to(device)
            label.squeeze_()
            output = model(x)
            _, prediction = torch.max(output.data, 1)
            correct += torch.sum(prediction == label.data)
            total += label.size(0)
    acc = 100. * correct / total
    print('test epoch:{} accuracy:{}'.format(epoch,acc))
    test_result.write('test epoch:{} accuracy:{}'.format(epoch,acc))
    test_result.write('\n')
    test_result.close()


epoch = 200
model = resnet18().to(device)

train_root='train/'
test_root='test/'
train_loader, test_loader = obtain_loader(train_root,test_root)
# 优化器和学习率衰减
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.8, last_epoch=-1)
# 保存模型参数
checkpoint = torch.load('model3.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])


for epoch in range(151,epoch):
    scheduler.step()
    train_loop(model, train_loader, optimizer,epoch)
    test_loop(model, test_loader,epoch)

    if epoch == 200:
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, 'model4.pth')
        break
