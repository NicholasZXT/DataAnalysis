# coding:utf8
import os
import ipdb
import torch as t
import torchvision as tv
import tqdm
from model import NetG, NetD
from visualize import Visualizer
# 这个包有问题，github上改名成了 torchtnt，里面也找不到 meters 模块和里面的 AverageValueMeter
# from torchnet.meters import AverageValueMeter
# AverageValueMeter 就是一个简单统计均值的封装类


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

    debug_file = 'debuggan'  # 当前目录下存在该文件，则利用 ipdb 进入debug模式
    d_every = 1  # 每1个batch训练一次判别器
    g_every = 5  # 每5个batch训练一次生成器
    save_every = 10  # 每10个epoch保存一次模型
    netd_path = None  # 'checkpoints/netd_.pth' #预训练模型
    netg_path = None  # 'checkpoints/netg_211.pth'

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

    device = t.device('cuda') if opt.gpu else t.device('cpu')
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
    netg, netd = NetG(opt), NetD(opt)
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)

    # 定义优化器和损失
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss().to(device)

    # 真图片label为1，假图片label为0
    # noises为生成网络的输入
    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.zeros(opt.batch_size).to(device)
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    # errord_meter = AverageValueMeter()
    # errorg_meter = AverageValueMeter()

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
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises).detach()  # 根据噪声生成假图
                output = netd(fake_img)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()
                optimizer_d.step()

                error_d = error_d_fake + error_d_real
                # errord_meter.add(error_d.item())

            if ii % opt.g_every == 0:
                print(f"[epoch {epoch + 1}][batch {ii+1}]: training generator ...")
                optimizer_g.zero_grad()
                # 训练生成器
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                # errorg_meter.add(error_g.item())

            if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
                print(f"[epoch {epoch + 1}][batch {ii+1}]: visualizing ...")
                # 可视化
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                fix_fake_imgs = netg(fix_noises)
                vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                # vis.plot('errord', errord_meter.value()[0])
                # vis.plot('errorg', errorg_meter.value()[0])

        if (epoch + 1) % opt.save_every == 0:
            print(f"[epoch {epoch + 1}]: saving checkpoint ...")
            # 保存模型、图片
            if fix_fake_imgs:
                tv.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, epoch),
                                    normalize=True, range=(-1, 1))
            t.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
            t.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % epoch)
            # errord_meter.reset()
            # errorg_meter.reset()


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
    train(data_path=data_dir, save_path='result_images', max_epoch=15, batch_size=256*4, save_every=5)


