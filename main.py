import argparse
from PIL import Image

from utils.prepare_images import *
from utils.transform import trans
from Models import *

import torch


def get_model():
    checkpoint = "model_check_point/CRAN_V2/CARN_model_checkpoint.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))
    model_cran_v2 = network_to_half(model_cran_v2)
    model_cran_v2.load_state_dict(torch.load(checkpoint, device))
    model_cran_v2 = model_cran_v2.float()

    return model_cran_v2

def scale2(img):
    img = img.resize((img.size[0] * 2, img.size[1] * 2))
    img = img.convert("RGB")
    print('Type image', type(img))
    img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.BICUBIC) 
    img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
    img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
    with torch.no_grad():
        out = [model(i) for i in img_patches]
    img_upscale = img_splitter.merge_img_tensor(out)
    print('Upscale type: ', img_upscale.shape)
    out_image = trans(img_upscale)
    return out_image
    
def scale(img, sf):
    print(sf)
    x = 1
    while x < sf:
        print(x)
        img = scale2(img)
        x *= 2
    img = img.resize((img.size[0] * sf // x, img.size[1] * sf // x))
    return img

if __name__ == '__main__':
    model = get_model()
    
    parser = argparse.ArgumentParser(description='Scale up image')
    parser.add_argument('--input', type=str, help='Input image to scale', required=True)
    parser.add_argument('--output', type=str, help='Output image', required=True)
    parser.add_argument('--scale', type=int, help='Scale factor', default=2)
    parser.add_argument('--key', type=str,help='Key to run program',required=False)

    opt = parser.parse_args()

    #if (opt.key != 'qwerQWE@@'):
    #    raise Exception("Product key is invalid. Please check your key")

    if opt.scale > 16:
        raise Exception("Scale factor is too large, choose a smaller one")

    img = Image.open(opt.input)


    out = scale(img, opt.scale)    
    out = convert_to_png(out)
    out.save(opt.output)
    print("OKK")
