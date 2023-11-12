# coding:utf8
import os
import ipdb
import torch as t
import torchvision as tv
import tqdm
from datetime import datetime, timedelta
from model import NetG, NetD
from visualize import Visualizer
# 这个包有问题，github上改名成了 torchtnt，里面也找不到 meters 模块和对应的 AverageValueMeter 类
# from torchnet.meters import AverageValueMeter
import math
import numpy as np


class AverageValueMeter:
    """
    AverageValueMeter 就是一个简单统计均值和方差的封装类。
    从https://tnt.readthedocs.io/en/latest/_modules/torchnet/meter/averagevaluemeter.html#AverageValueMeter复制过来的实现代码
    """
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean, self.std = self.sum, np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = math.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


class Config:
    num_workers = 4  # 多进程加载数据所用的进程数
    data_path = 'data/'  # 数据集存放路径，里面要求只有一个faces文件夹，存放所有的头像数据
    save_path = 'imgs/'  # 生成图片保存路径

    image_size = 96  # 图片尺寸
    batch_size = 256
    max_epoch = 200
    lr1 = 2e-4  # 生成器的学习率
    lr2 = 2e-4  # 判别器的学习率
    beta1 = 0.5  # Adam优化器的beta1参数
    gpu = True  # 是否使用GPU
    nz = 100  # 噪声维度
    ngf = 64  # 生成器feature map数
    ndf = 64  # 判别器feature map数

    vis = False  # 是否使用visdom可视化
    env = 'GAN'  # visdom的env
    plot_every = 20  # 每间隔 20 batch，visdom画图一次

    # 一般说来，判别器要比生成器更新的频率要高 ----- KEY
    d_every = 1  # 每1个batch训练一次判别器
    g_every = 5  # 每5个batch训练一次生成器
    save_every = 10  # 每10个epoch保存一次模型
    netd_path = None  # 'checkpoints/netd_.pth' #预训练模型
    netg_path = None  # 'checkpoints/netg_211.pth'

    debug_file = 'debuggan'  # 当前目录下存在该文件，则利用 ipdb 进入debug模式
    # 只测试不训练 —— 用于 generate 函数
    gen_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 噪声的均值
    gen_std = 1  # 噪声的方差


opt = Config()


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    if opt.vis:
        vis = Visualizer(opt.env)

    # 数据
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.num_workers,
                                         drop_last=True
                                         )

    # 网络
    device = t.device('cuda') if opt.gpu else t.device('cpu')
    netg, netd = NetG(opt), NetD(opt)
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        print(f"*** discriminator model loading checkpoint from: {opt.netd_path}")
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        print(f"*** generator model loading checkpoint from: {opt.netg_path}")
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)

    # 定义优化器和损失
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss().to(device)

    # 真图片label为1，假图片label为0
    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.zeros(opt.batch_size).to(device)
    # noises为生成网络的输入
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    # 记录判别器和生成器错误的类
    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()

    epochs = range(opt.max_epoch)
    fix_fake_imgs = None
    for epoch in iter(epochs):
        print(f"------ training epoch [{epoch+1}] ------")
        for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
            real_img = img.to(device)
            if ii % opt.d_every == 0:
                print(f"[epoch {epoch + 1}][batch {ii+1}]: training discriminator ...")
                optimizer_d.zero_grad()
                # 训练判别器
                # 尽可能的把真图片判别为正确
                output = netd(real_img)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()
                # 尽可能把假图片判别为错误
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))  # 这里重新进行了一次随机采样，生成噪声输入
                fake_img = netg(noises).detach()    # 生成器根据噪声生成假图
                output = netd(fake_img)             # 判别器再判断假图片
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()
                optimizer_d.step()
                error_d = error_d_fake + error_d_real
                errord_meter.add(error_d.item())

            if ii % opt.g_every == 0:
                print(f"[epoch {epoch + 1}][batch {ii+1}]: training generator ...")
                optimizer_g.zero_grad()
                # 训练生成器
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))  # 再次进行了一次随机采样，生成噪声输入
                fake_img = netg(noises)
                output = netd(fake_img)
                # 注意，这里输入生成器输出的output是噪声产生的假图片，但是true_labels却是真实图片的label
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                errorg_meter.add(error_g.item())

            if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
                print(f"[epoch {epoch + 1}][batch {ii+1}]: visualizing ...")
                # 可视化
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                fix_fake_imgs = netg(fix_noises)
                vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake', opts=dict(title='G-Fake images'))
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real', opts=dict(title='Real images'))
                vis.plot('D-error', errord_meter.value()[0])
                vis.plot('G-error', errorg_meter.value()[0])

        if (epoch + 1) % opt.save_every == 0:
            print(f"[epoch {epoch + 1}]: saving checkpoint ...")
            # 保存模型、图片
            if hasattr(fix_fake_imgs, 'data'):
                tv.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, epoch), normalize=True,
                                    range=(-1, 1))
            t.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
            t.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % epoch)
            errord_meter.reset()
            errorg_meter.reset()


@t.no_grad()
def generate(**kwargs):
    """
    随机生成动漫头像，并根据netd的分数选择较好的
    """
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda') if opt.gpu else t.device('cpu')
    netg, netd = NetG(opt).eval(), NetD(opt).eval()
    noises = t.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    noises = noises.to(device)

    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)

    # 生成图片，并计算图片在判别器的分数
    fake_img = netg(noises)
    scores = netd(fake_img).detach()

    # 挑选最好的某几张
    indexs = scores.topk(opt.gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_img.data[ii])
    # 保存图片
    tv.utils.save_image(t.stack(result), opt.gen_img, normalize=True, range=(-1, 1))


if __name__ == '__main__':
    # 这里不使用fire库
    # import fire
    # fire.Fire()
    print(os.getcwd())
    data_dir = r"D:\Project-Workspace\Python-Projects\DataAnalysis\local-datasets\GAN-facing\data"
    save_path = os.path.join(os.getcwd(), 'result_images')
    netg_path = os.path.join(os.getcwd(), 'checkpoints', 'netg_179.pth')
    netd_path = os.path.join(os.getcwd(), 'checkpoints', 'netd_179.pth')
    print(os.path.exists(save_path))
    start = datetime.now()
    print(f"*********** start training at {start.strftime('%Y-%m-%d %H:%M:%S')} ***********")
    # train(data_path=data_dir, num_workers=2, save_path=save_path, max_epoch=15, batch_size=256*4, save_every=5, vis=True)
    train(data_path=data_dir, num_workers=2, save_path=save_path, max_epoch=200, batch_size=256*3, save_every=20, vis=True)
    # train(data_path=data_dir, num_workers=2, save_path=save_path, max_epoch=200, batch_size=256*3, save_every=20,
    #       vis=True, netg_path=netg_path, netd_path=netd_path)
    end = datetime.now()
    print(f"*********** end training at {end.strftime('%Y-%m-%d %H:%M:%S')} ***********")
    print(f"training time in seconds: {(end-start).seconds}")
    # RTX 2070-Super: epoch 200, batch_size 256*3, 耗时约 5940 s
    # RTX 3060-12G: epoch 200, batch_size 256*3, 耗时约 4439 s
