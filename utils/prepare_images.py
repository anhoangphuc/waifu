import copy
import glob
import os
from multiprocessing.dummy import Pool as ThreadPool

from PIL import Image

from Models import *
import torch

import numpy as np

try:
    import accimage
except ImportError:
    accimage = None


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def _is_numpy(img):
    return isinstance(img, np.ndarray)


def _is_numpy_image(img):
    return img.ndim in {2, 3}

def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


class ImageSplitter:
    # key points:
    # Boarder padding and over-lapping img splitting to avoid the instability of edge value
    # Thanks Waifu2x's autorh nagadomi for suggestions (https://github.com/nagadomi/waifu2x/issues/238)

    def __init__(self, seg_size=48, scale_factor=2, boarder_pad_size=3):
        self.seg_size = seg_size
        self.scale_factor = scale_factor
        self.pad_size = boarder_pad_size
        self.height = 0
        self.width = 0
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear')

    def split_img_tensor(self, pil_img, scale_method=Image.BILINEAR, img_pad=0):
        # resize image and convert them into tensor
        img_tensor = to_tensor(pil_img).unsqueeze(0)
        img_tensor = nn.ReplicationPad2d(self.pad_size)(img_tensor)
        batch, channel, height, width = img_tensor.size()
        self.height = height
        self.width = width

        if scale_method is not None:
            img_up = pil_img.resize((2 * pil_img.size[0], 2 * pil_img.size[1]), scale_method)
            img_up = to_tensor(img_up).unsqueeze(0)
            img_up = nn.ReplicationPad2d(self.pad_size * self.scale_factor)(img_up)

        patch_box = []
        # avoid the residual part is smaller than the padded size
        if height % self.seg_size < self.pad_size or width % self.seg_size < self.pad_size:
            self.seg_size += self.scale_factor * self.pad_size

        # split image into over-lapping pieces
        for i in range(self.pad_size, height, self.seg_size):
            for j in range(self.pad_size, width, self.seg_size):
                part = img_tensor[:, :,
                       (i - self.pad_size):min(i + self.pad_size + self.seg_size, height),
                       (j - self.pad_size):min(j + self.pad_size + self.seg_size, width)]
                if img_pad > 0:
                    part = nn.ZeroPad2d(img_pad)(part)
                if scale_method is not None:
                    # part_up = self.upsampler(part)
                    part_up = img_up[:, :,
                              self.scale_factor * (i - self.pad_size):min(i + self.pad_size + self.seg_size,
                                                                          height) * self.scale_factor,
                              self.scale_factor * (j - self.pad_size):min(j + self.pad_size + self.seg_size,
                                                                          width) * self.scale_factor]

                    patch_box.append((part, part_up))
                else:
                    patch_box.append(part)
        return patch_box

    def merge_img_tensor(self, list_img_tensor):
        out = torch.zeros((1, 3, self.height * self.scale_factor, self.width * self.scale_factor))
        img_tensors = copy.copy(list_img_tensor)
        rem = self.pad_size * 2

        pad_size = self.scale_factor * self.pad_size
        seg_size = self.scale_factor * self.seg_size
        height = self.scale_factor * self.height
        width = self.scale_factor * self.width
        for i in range(pad_size, height, seg_size):
            for j in range(pad_size, width, seg_size):
                part = img_tensors.pop(0)
                part = part[:, :, rem:-rem, rem:-rem]
                # might have error
                if len(part.size()) > 3:
                    _, _, p_h, p_w = part.size()
                    out[:, :, i:i + p_h, j:j + p_w] = part
                # out[:,:,
                # self.scale_factor*i:self.scale_factor*i+p_h,
                # self.scale_factor*j:self.scale_factor*j+p_w] = part
        out = out[:, :, rem:-rem, rem:-rem]
        return out


def load_single_image(img_file,
                      up_scale=False,
                      up_scale_factor=2,
                      up_scale_method=Image.BILINEAR,
                      zero_padding=False):
    img = Image.open(img_file).convert("RGB")
    out = to_tensor(img).unsqueeze(0)
    if zero_padding:
        out = nn.ZeroPad2d(zero_padding)(out)
    if up_scale:
        size = tuple(map(lambda x: x * up_scale_factor, img.size))
        img_up = img.resize(size, up_scale_method)
        img_up = to_tensor(img_up).unsqueeze(0)
        out = (out, img_up)

    return out


def standardize_img_format(img_folder):
    def process(img_file):
        img_path = os.path.dirname(img_file)
        img_name, _ = os.path.basename(img_file).split(".")
        out = os.path.join(img_path, img_name + ".JPEG")
        os.rename(img_file, out)

    list_imgs = []
    for i in ['png', "jpeg", 'jpg']:
        list_imgs.extend(glob.glob(img_folder + "**/*." + i, recursive=True))
    print("Found {} images.".format(len(list_imgs)))
    pool = ThreadPool(4)
    pool.map(process, list_imgs)
    pool.close()
    pool.join()

def convert_to_png(img):
    img_num = np.asarray(img.convert("RGBA"))
    nn = img_num.copy()
    nn[nn[:, :, 0:3].sum(axis=2) <= 10] = [0, 0, 0, 0]
    return Image.fromarray(nn)
