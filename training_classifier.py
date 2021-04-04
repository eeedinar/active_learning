import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
# import human_annotator
import torch.nn as nn
import torch.nn.functional as F
from active_learning import *
from torchviz import make_dot
# torch.manual_seed(0)
# np.random.seed(0)

def normalize(tensor):
    tensor = tensor.clone()  # avoid modifying tensor in-place
    if range is not None:
        assert isinstance(range, tuple), \
            "range has to be a tuple (min, max) if specified. min and max are numbers"

    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)

    def norm_range(t, range):
        if range is not None:
            norm_ip(t, range[0], range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    # if scale_each is True:
    #     for t in tensor:  # loop over mini-batch dimension
    #         norm_range(t, range)
    # else:
    return norm_range(tensor, range)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

def train_model(net,trainloader,testloader,epochs):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
    
    print('Finished Training')
    
    # PATH = './cifar_net.pth'
    # torch.save(net.state_dict(), PATH)
    
    # net.load_state_dict(torch.load(PATH))
    # outputs = net(images)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))
    
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(2):
        print('Accuracy of %5s : %4f %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    return net

def load_data(batch_size, dataroot, original_cifar):
    testset = torchvision.datasets.ImageFolder(root=dataroot+"test/",
                               transform=transforms.Compose([
                                   # transforms.Resize(image_size),
                                   # transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)
    if original_cifar:
        trainset = torchvision.datasets.ImageFolder(root=dataroot+"generated_images/",
                                   transform=transforms.Compose([
                                       # transforms.Resize(image_size),
                                       # transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        trainset = torchvision.datasets.ImageFolder(root=dataroot+"generated_images/",
                                   transform=transforms.Compose([
                                       # transforms.Resize(image_size),
                                       # transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    return trainloader, testloader


# def choose_subset(cifar_subset):
#     newpath = r'cifar_dataset_AL_temp/AL_training' 
#     if not os.path.exists(newpath):
#         os.makedirs(newpath)
#     if not os.path.exists(newpath+'/cat'):
#         os.makedirs(newpath)
    
use_gpu = True
device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu) else "cpu")
dataroot = 'cifar_dataset_AL_temp/'
cifar_subset = 100
batch_size = 10
# dataroot = 'cifar10_64_64/'
epoch = 35
classes = ('plane', 'cat')


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


net = Net()

make_dot(y.mean(), params=dict(net.named_parameters()))
torch.save(net.state_dict(), 'learner.pth')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


trainloader, testloader = load_data(batch_size, dataroot, original_cifar=True) 
net = train_model(net,trainloader,testloader,epoch)

    
#Gan part

# Generator Code

# Batch size during training
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

netG = Generator(1).to(device)
netG.load_state_dict(torch.load('netG_10.pth'))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
fake_images = netG(fixed_noise).detach().cpu()
fake_images_grid = torchvision.utils.make_grid(fake_images, padding=2, normalize=True)

plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(fake_images_grid, (1,2,0)))
plt.savefig('example_gan_out_2.png', bbox_inches='tight', dpi=500) 
plt.show()

num_of_classes = 2
# # print(calc_entropy(predictions,num_of_classes))
# choose_sample(fake_outputs,num_of_classes,'entropy')
# # choose_sample(fake_outputs,num_of_classes,'least confident')
# choose_sample(fake_outputs,num_of_classes,'margin sampling')



active_learning_cycle = 3
images_per_cycle = 20
num_of_selection = 10
for cycle in range(active_learning_cycle):
    print('Active learning cycle' , cycle+1 , " / ", active_learning_cycle)
    latent_vars = torch.randn(images_per_cycle, nz, 1, 1, device=device)
    fake_images = netG(latent_vars).detach().cpu()
    fake_outputs = net(fake_images).detach().cpu().numpy()
    
    # selected_samples = choose_sample(fake_outputs,num_of_classes,'entropy',num_of_selection)
    selected_samples = choose_sample(fake_outputs,num_of_classes,'least confident',num_of_selection)
    for image_id in selected_samples:
        torchvision.utils.save_image(fake_images[image_id,:,:,:], 'C:/Users/ahmet/Desktop/EECE7370/Project/cifar_dataset_AL_temp/generated_images/img_{}.png'.format(image_id), normalize=True)
    exec(open("human_annotator.py").read())
    trainloader, testloader = load_data(batch_size, dataroot, original_cifar=False) 
    net = train_model(net,trainloader,testloader,epoch)
    # net = Net()    
    # net.load_state_dict(torch.load('learner.pth'))

    
