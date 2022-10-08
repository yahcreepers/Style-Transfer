import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from random import choices, sample, choice
from torchvision.transforms import Compose, RandomHorizontalFlip, Normalize, ToTensor

it = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
st = ToTensor()

class Mydataset( Dataset ) :
    def __init__( self, mode, custom_len = None, img_transform = it, style_transform = st ) :
        super().__init__()
        self.img_list = os.listdir( '/tmp2/PAYToWin/PM/Dataset/photo_jpg' )
        self.img_list = [ '/tmp2/PAYToWin/PM/Dataset/photo_jpg/' + i for i in self.img_list ]
        self.style_list = os.listdir( '/tmp2/PAYToWin/PM/Dataset/monet_jpg/' )
        self.style_list = [ '/tmp2/PAYToWin/PM/Dataset/monet_jpg/' + i for i in self.style_list ]
        self.mode = mode
        if mode == 'pad':
            self.style_list = choices( self.style_list, k = len( self.img_list ) )
        elif mode == 'truncate' :
            self.img_list = sample( self.img_list, len( self.style_list ) )
        elif mode == 'custom' :
            self.img_list = sample( self.img_list, custom_len )
            self.style_list = choices( self.style_list, k = custom_len )
        elif mode == 'random':
            pass
        else:
            print( "invalid mode" )
            exit()
        self.img_transform = img_transform
        self.style_transform = style_transform
    
    def __len__( self ):
        return len( self.img_list )
    
    def __getitem__( self, idx ) :
        if self.mode == 'random':
            img = Image.open( choice(self.img_list) )
            style = Image.open( choice( self.style_list ) )
        else :
            img = Image.open( self.img_list[idx] )
            style = Image.open( self.style_list[idx] )
        img = self.img_transform( img )
        style = self.style_transform( style )
        return img, style
