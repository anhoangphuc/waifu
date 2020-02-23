import argparse
from PIL import Image

from utils.prepare_images import *
from Models import *

from torchvision.utils import save_image


def get_model():
    checkpoint = "model_check_point/CRAN_V2/CARN_model_checkpoint.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))
    model_cran_v2 = network_to_half(model_cran_v2)
    print('device', device)
    model_cran_v2.load_state_dict(torch.load(checkpoint, device))
    model_cran_v2 = model_cran_v2.float()

    return model_cran_v2

def scale2(img):
    img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.BICUBIC) 
    img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
    img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
    with torch.no_grad():
        out = [model(i) for i in img_patches]
    img_upscale = img_splitter.merge_img_tensor(out)
    print('Upscale type: ', type(img_upscale))
    return img_upscale
    

if __name__ == '__main__':
    model = get_model()
    
    parser = argparse.ArgumentParser(description='Scale up image')
    parser.add_argument('--input', type=str, help='Input image to scale', required=True)
    parser.add_argument('--output', type=str, help='Output image', required=True)
    parser.add_argument('--scale', type=int, help='Scale factor', default=2)

    opt = parser.parse_args()

    demo_img = opt.input
    img = Image.open(demo_img)
    img = img.resize((img.size[0] * opt.scale, img.size[1] * opt.scale))
    img = img.convert("RGB")

    img_t = to_tensor(img).unsqueeze(0) 
    save_image(img_t, 'original.jpg')

    #img = img.resize((img.size[0] // 2, img.size[1] // 2), Image.BICUBIC) 

    #img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
    #img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
    #with torch.no_grad():
    #    out = [model(i) for i in img_patches]
    #img_upscale = img_splitter.merge_img_tensor(out)
    
    out1 = scale2(img)
    save_image(out1, opt.output)
    #out2 = scale2(out1)
    #save_image(out2, opt.output)
    #final = torch.cat([img_t, img_upscale])
    #save_image(final, f'compare{opt.output}', nrow=2)

    print("OKK")
