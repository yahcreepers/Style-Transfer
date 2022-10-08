import torch
import os
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from dataset import Mydataset
from model import *
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from PIL import Image
from skimage import io, transform
import itertools


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
normalCrop = transforms.Compose([transforms.RandomCrop(256),
                            transforms.ToTensor(),
                            normalize])

Crop = transforms.Compose([transforms.RandomCrop(256),
                            transforms.ToTensor()])

generateNCrop = transforms.Compose([transforms.CenterCrop(512),
                            transforms.ToTensor(),
                            normalize])

generateCrop = transforms.Compose([transforms.CenterCrop(512),
                            transforms.ToTensor()])

it = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    normalize
])
st = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).cuda(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).cuda(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def Cut(image_path, saving_path):
    image = io.imread(image_path)
    H, W, _ = image.shape
    if H > W:
        H = int(H/W*512)
        W = 512
    else:
        W = int(W/H*512)
        H = 512
    image = transform.resize(image, (H, W), mode="reflect", anti_aliasing=True)
    io.imsave(saving_path, image)
    image = Image.open(saving_path)
    return image

def main(args):
    
    if args.do_train:
        if args.max_train_samples != None:
            train_dataset = Mydataset(mode="custom", custom_len=args.max_train_samples, img_transform = it, style_transform = st)
        else:
            train_dataset = Mydataset(mode="pad", img_transform = it, style_transform = st)
        #test_dataset = Image_Dataset(args.test_content_dir,args.test_style_dir)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        #test_dataloader = DataLoader(test_dataset, batch_size=args.batch)
        valid_dataset = Mydataset(mode="custom", custom_len=100, img_transform = it, style_transform = st)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        
        c = args.cuda
        G_A2B = Generator().cuda(c)
        G_B2A = Generator().cuda(c)
        G_A2B.apply(weights_init)
        G_B2A.apply(weights_init)
        DA = Discriminator().cuda(c)
        DB = Discriminator().cuda(c)
#        DA = Discriminator_P(3, 32).cuda(c)
#        DB = Discriminator_P(3, 32).cuda(c)
        optimizerG = Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=args.glr)
        optimizerDA = Adam(DA.parameters(), lr=args.dlr)
        optimizerDB = Adam(DB.parameters(), lr=args.dlr)
        l1 = nn.L1Loss().cuda(c)
        l2 = nn.MSELoss().cuda(c)
        DA.eval()
        DB.eval()
        for e in tqdm(range(args.epoch)):
            for i, (content, style) in enumerate(train_dataloader):
                content = content.cuda(c)
                style = style.cuda(c)
                
                optimizerG.zero_grad()
                
                #identity loss
                same_A = G_B2A(content)
                same_B = G_A2B(style)
                identity_loss = (l1(same_A, content) + l1(same_B, style)) * 10
                
                #GAN loss
                B_fake = G_A2B(content)
                A_rec = G_B2A(B_fake)
                A_fake = G_B2A(style)
                B_rec = G_A2B(A_fake)
                
                DA_fake = DA(A_fake).detach()
                DB_fake = DB(B_fake).detach()
                GAN_loss = (l2(DA_fake, torch.ones(DA_fake.shape[0]).cuda(c).view(-1, 1)) + l2(DB_fake, torch.ones(DB_fake.shape[0]).cuda(c).view(-1, 1)))
                
                #Cycle loss
                Cycle_loss = (l1(A_rec, content) + l1(B_rec, style)) * 10
                
                loss_G = identity_loss + GAN_loss + Cycle_loss
                loss_G.backward()
                optimizerG.step()
                
                #Discriminator loss
                optimizerDA.zero_grad()
                A_pred = DA(A_fake.detach())
                A_real = DA(content)
                
                DA_loss = (l2(A_pred, torch.zeros(A_pred.shape[0]).cuda(c).view(-1, 1)) + l2(A_real, torch.ones(A_real.shape[0]).cuda(c).view(-1, 1))) * 0.5
                DA_loss.backward()
                optimizerDA.step()
                
                optimizerDB.zero_grad()
                B_pred = DB(B_fake.detach())
                B_real = DB(style)
                
                DB_loss = (l2(B_pred, torch.zeros(B_pred.shape[0]).cuda(c).view(-1, 1)) + l2(B_real, torch.ones(B_real.shape[0]).cuda(c).view(-1, 1))) * 0.5
                DB_loss.backward()
                optimizerDB.step()
                if i % args.saving_step == 0 and i != 0:
                    test = [torch.FloatTensor([]).cuda(c) for j in range(4)]
                    print(f"identity_loss = {identity_loss}, GAN_loss = {GAN_loss}, Cycle_loss = {Cycle_loss}, DA_loss = {DA_loss}, DB_loss = {DB_loss}")
                    for j, (content, style) in enumerate(valid_dataloader):
                        content = content.cuda(c)
                        style = style.cuda(c)
                        with torch.no_grad():
                            A_G = G_A2B(content)
                            A_origin = G_B2A(A_G)
                            B_G = G_B2A(style)
                            B_origin = G_A2B(B_G)
                        test[0] = torch.cat((test[0], denorm(A_G, c)), 0)
                        test[1] = torch.cat((test[1], denorm(B_G, c)), 0)
                        test[2] = torch.cat((test[2], denorm(A_origin, c)), 0)
                        test[3] = torch.cat((test[3], denorm(B_origin, c)), 0)
                    #print(test[0].shape)
                    save_image(test[0], f'{args.output_dir}/check-{e}_A_G.png', nrow=10)
                    save_image(test[1], f'{args.output_dir}/check-{e}_B_G.png', nrow=10)
                    save_image(test[2], f'{args.output_dir}/check-{e}_A_origin.png', nrow=10)
                    save_image(test[3], f'{args.output_dir}/check-{e}_B_origin.png', nrow=10)
            
            if e % args.saving_epoch == 0 and e != 0:
                torch.save(G_A2B.state_dict(), f'{args.output_dir}/check-{e}_G_A2B.pth')
                torch.save(G_B2A.state_dict(), f'{args.output_dir}/check-{e}_G_B2A.pth')
                torch.save(DA.state_dict(), f'{args.output_dir}/check-{e}_DA.pth')
                torch.save(DB.state_dict(), f'{args.output_dir}/check-{e}_DB.pth')
                
                
            
            
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./test/")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--train_content_dir", type=str, default="")
    parser.add_argument("--test_content_dir", type=str, default="")
    parser.add_argument("--train_style_dir", type=str, default="")
    parser.add_argument("--test_style_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--saving_step", type=int, default=10000)
    parser.add_argument("--saving_epoch", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--glr", type=float, default=1e-4)
    parser.add_argument("--dlr", type=float, default=1e-4)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model", type=str, default="vgg")
    parser.add_argument("--lamb", type=float, default=10)
    
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    main(args)

