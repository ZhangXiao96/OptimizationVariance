from lib.ModelWrapper import ModelWrapper
from tensorboardX import SummaryWriter
import torch
from torchvision import transforms, datasets
import numpy as np
import random
import sys
import os

args = sys.argv
data_name = args[1]     # 'svhn', 'cifar10', 'cifar100'
model_name = args[2]    # 'resnet18', 'resnet34', 'vgg16', 'vgg13', 'vgg11'
noise_split = float(args[3])
opt = args[4]
lr = float(args[5])
test_id = int(args[6])
data_root = args[7]

# setting
train_batch_size = 128
train_epoch = 250
eval_batch_size = 250

data_root = os.path.join(data_root, data_name)
if data_name == 'cifar10':
    dataset = datasets.CIFAR10
    nb_class = 10
    from archs.cifar10 import vgg, resnet
elif data_name == 'cifar100':
    dataset = datasets.CIFAR100
    nb_class = 100
    from archs.cifar100 import vgg, resnet
elif data_name == 'svhn':
    dataset = datasets.SVHN
    nb_class = 10
    from archs.svhn import vgg, resnet
else:
    raise Exception('No such dataset')

if model_name == 'vgg11':
    model = vgg.vgg11_bn()
elif model_name == 'vgg13':
    model = vgg.vgg13_bn()
elif model_name == 'vgg16':
    model = vgg.vgg16_bn()
elif model_name == 'resnet18':
    model = resnet.resnet18()
elif model_name == 'resnet34':
    model = resnet.resnet34()
else:
    raise Exception("No such model!")

train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
eval_transform = transforms.Compose([transforms.ToTensor()])

# load data
if 'cifar' in data_name:
    train_data = dataset(data_root, train=True, transform=train_transform, download=True)
    if noise_split > 0:
        train_targets = np.array(train_data.targets)
        data_size = len(train_targets)
        random_index = random.sample(range(data_size), int(data_size*noise_split))
        random_part = train_targets[random_index]
        np.random.shuffle(random_part)
        train_targets[random_index] = random_part
        train_data.targets = train_targets.tolist()

        noise_data = dataset(data_root, train=False, transform=eval_transform, download=True)
        noise_data.targets = random_part.tolist()
        noise_data.data = train_data.data[random_index]

    test_data = dataset(data_root, train=False, transform=eval_transform)
    var_data = dataset(data_root, train=True, transform=eval_transform, download=True)

elif 'svhn' in data_name:
    train_data = dataset(data_root, split='train', transform=train_transform, download=True)
    if noise_split > 0:
        train_targets = np.array(train_data.labels)
        data_size = len(train_targets)
        random_index = random.sample(range(data_size), int(data_size * noise_split))
        random_part = train_targets[random_index]
        np.random.shuffle(random_part)
        train_targets[random_index] = random_part
        train_data.labels = train_targets.tolist()

        noise_data = dataset(data_root, split='test', transform=eval_transform, download=True)
        noise_data.labels = random_part.tolist()
        noise_data.data = train_data.data[random_index]
    test_data = dataset(data_root, split='test', transform=eval_transform)
    var_data = dataset(data_root, split='train', transform=eval_transform, download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=0,
                                           drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=0,
                                          drop_last=False)
var_loader = torch.utils.data.DataLoader(var_data, batch_size=train_batch_size, shuffle=False, num_workers=0,
                                          drop_last=False)

if noise_split > 0:
    noise_loader = torch.utils.data.DataLoader(noise_data, batch_size=eval_batch_size, shuffle=True, num_workers=0,
                                               drop_last=False)

# build model
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

if opt == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif opt == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

wrapper = ModelWrapper(model, optimizer, criterion, device)

# train the model
save_path = os.path.join('runs', 'noise_{}_opt_{}_lr_{}'.format(noise_split, opt, lr),
                         data_name, "{}".format(model_name), "{}".format(test_id))
if not os.path.exists(save_path):
    os.makedirs(save_path)
writer = SummaryWriter(log_dir=os.path.join(save_path, "log"), flush_secs=30)

wrapper.train()
for id_epoch in range(train_epoch):
    # train loop
    train_loss = 0
    train_acc = 0
    train_size = 0

    for id_batch, (inputs, targets) in enumerate(train_loader):
        loss, acc, correct, _, _ = wrapper.train_on_batch_with_gradients_recorded(inputs, targets)


        train_loss += loss
        train_acc += correct
        train_size += len(targets)
        print("epoch:{}/{}, batch:{}/{}, loss={}, acc={}".
              format(id_epoch + 1, train_epoch, id_batch + 1, len(train_loader), loss, acc))

    # recorder loss and acc
    train_loss /= id_batch + 1
    train_acc /= train_size
    writer.add_scalar("train acc", train_acc, id_epoch+1)
    writer.add_scalar("train loss", train_loss, id_epoch+1)

    # recorder output var
    wrapper.eval()
    optimization_var = wrapper.get_optimization_var(var_loader)
    writer.add_scalar("optimization var", train_var, id_epoch+1)

    # eval
    wrapper.eval()
    test_loss, test_acc = wrapper.eval_all(test_loader)
    print("epoch:{}/{}, batch:{}/{}, testing...".format(id_epoch + 1, train_epoch, id_batch + 1, len(train_loader)))
    print("clean: loss={}, acc={}".format(test_loss, test_acc))
    writer.add_scalar("test acc", test_acc, id_epoch+1)
    writer.add_scalar("test loss", test_loss, id_epoch+1)
    if noise_split > 0:
        noise_loss, noise_acc = wrapper.eval_all(noise_loader)
        print("noise: loss={}, acc={}".format(noise_loss, noise_acc))
        writer.add_scalar("noise acc", noise_acc, id_epoch+1)
        writer.add_scalar("noise loss", noise_loss, id_epoch+1)

    wrapper.train()
writer.close()
