import torch
import os
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from dataset import Mydataset
from model import AdaIN_Model
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from PIL import Image
from skimage import io, transform


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
        
        c = args.cuda
        model = AdaIN_Model(args.model)
        model = model.cuda(c)
        optimizer = Adam(model.parameters(), lr=args.lr)
        for e in range(args.epoch):
            for i, (content, style) in tqdm(enumerate(train_dataloader)):
                content = content.cuda(c)
                style = style.cuda(c)
                content_loss, style_loss = model(content, style)
                total_loss = content_loss + args.lamb * style_loss
                if i % args.saving_step == 0 and i != 0:
                    print(content_loss.item(), style_loss.item(), total_loss.item())
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                if i % args.saving_step == 0 and i != 0:
                    with torch.no_grad():
                        origin = model.origin(content)
                    with torch.no_grad():
                        output = model.generate(content, style)
                    if args.model == "vgg":
                        content = denorm(content, c)
                        style = denorm(style, c)
                        output = denorm(output, c)
                        origin = denorm(origin, c)
                    fusion = torch.cat([content, style, output, origin], dim=0)
                    fusion = fusion.cuda(c)
                    save_image(fusion, f'{args.output_dir}/check-{e}_generate.png', nrow=args.batch_size)
            if e % args.saving_epoch == 0 and e != 0:
                torch.save(model.state_dict(), f'{args.output_dir}/check-{e}_model.pth')
    if args.do_predict:
        c = args.cuda
        model = AdaIN_Model(args.model)
        model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage))
        model = model.cuda(c)
        content_files = os.listdir(args.test_content_dir)
        style_files = os.listdir(args.test_style_dir)
        for content_file in content_files:
            content = Cut(f"{args.test_content_dir}/{content_file}", "./content_img.png")
            if args.model == "vgg":
                content_tensor = generateNCrop(content).unsqueeze(0).cuda(c)
            if args.model == "dense":
                content_tensor = generateCrop(content).unsqueeze(0).cuda(c)
            with torch.no_grad():
                output = model.origin(content_tensor)
            if args.model == "vgg":
                output = denorm(output, c)
            save_image(output, f"{args.output_dir}/{content_file}_origin.png", nrow=1)
            for style_file in style_files:
                content = Cut(f"{args.test_content_dir}/{content_file}", f"{args.output_dir}/{content_file}")
                style = Cut(f"{args.test_style_dir}/{style_file}", f"{args.output_dir}/{style_file}")
                if args.model == "vgg":
                    content_tensor = generateNCrop(content).unsqueeze(0).cuda(c)
                    style_tensor = generateNCrop(style).unsqueeze(0).cuda(c)
                if args.model == "dense":
                    content_tensor = generateCrop(content).unsqueeze(0).cuda(c)
                    style_tensor = generateCrop(style).unsqueeze(0).cuda(c)
                with torch.no_grad():
                    output = model.generate(content_tensor, style_tensor)
                if args.model == "vgg":
                    output = denorm(output, c)
                save_image(content_tensor, f"{args.output_dir}/{content_file}", nrow=1)
                save_image(style_tensor, f"{args.output_dir}/{style_file}", nrow=1)
                save_image(output, f"{args.output_dir}/{content_file}_{style_file}.png", nrow=1)
            
            
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
    parser.add_argument("--lr", type=float, default=3e-6)
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
