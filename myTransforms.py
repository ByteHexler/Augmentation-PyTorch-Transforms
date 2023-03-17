from numbers import Number
import random
from collections.abc import Sequence

import torch

from torchvision.transforms import functional as F
from torchvision.utils import _log_api_usage_once
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int

import numpy as np
from PIL import Image, ImageFilter
import cv2

from torchvision.transforms.transforms import *
from torchvision.transforms.transforms import __all__

__all__ = __all__ + ["HEDJitter", "RandomChoiceRotation", "RandomGaussBlur", "RandomGaussNoise", "RandomAffineCV2", "RandomElastic"]

rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29],
                         [0.07, 0.99, 0.11],
                         [0.27, 0.57, 0.78]])
hed_from_rgb = torch.linalg.inv(rgb_from_hed)

class HEDJitter(object):
    """Randomly perturbe the HED color space value an RGB image.
    First, it disentangled the hematoxylin and eosin color channels by color deconvolution method using a fixed matrix.
    Second, it perturbed the hematoxylin, eosin and DAB stains independently.
    Third, it transformed the resulting stains into regular RGB color space.
    Args:
        theta (float): How much to jitter HED color space,
         alpha is chosen from a uniform distribution [1-theta, 1+theta]
         beta is chosen from a uniform distribution [-theta, theta]
         the jitter formula is: s' = alpha * s + beta
        mode (str): specify which channels to jitter, either ``HE`` or ``HED``
    """
    def __init__(self, theta=0., mode='HE'): # HED_light: theta=0.05; HED_strong: theta=0.2
        _log_api_usage_once(self)
        assert isinstance(theta, Number), "theta should be a single number."
        assert mode in ['HE','HED'], "mode has to be either ``HE`` or ``HED``"
        self.theta = theta
        self.mode = mode

    @staticmethod
    def adjust_HED(img, theta, mode):

        alpha = torch.empty((3, 1)).uniform_(1-theta, 1+theta).to(img.device)
        beta = torch.empty((3, 1)).uniform_(-theta, theta).to(img.device)
        
        if mode=='HE':
            alpha[2,0]=1    # don't jitter D-channel
            beta[2,0]=0
        
        s = torch.matmul(-hed_from_rgb.to(img.device), torch.log(img.reshape((img.shape[0],3, -1)) + 1E-6))
        ns = alpha * s + beta  # perturbations on HED color space
        nimg = (torch.exp(torch.matmul(-rgb_from_hed.to(img.device), ns)) - 1E-6).view(img.shape) # np.reshape(color.rgb2hed(ns), img.shape) # 
        return nimg.clamp_(0,1)

    def __call__(self, img):
        return self.adjust_HED(img, self.theta, self.mode)

    def __repr__(self):
        return self.__class__.__name__+'(theta={0})'.format(self.theta)


class RandomChoiceRotation(object):
    """randomly select one of the provided angles for rotating the image (default: [0, 90, 180, 270]).

    Args:
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image[.Resampling].NEAREST``) are still accepted,
            but deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self, degrees=[0, 90, 180, 270],
        interpolation=InterpolationMode.NEAREST,
        expand=False,
        center=None,
        fill=None
    ):
        
        _log_api_usage_once(self)
        if isinstance(degrees, Sequence):
            for d in degrees:
                assert isinstance(d, Number), 'all elements of degree have to be numbers'
        else:
            assert isinstance(degrees, Number), 'degree has to be a number or a sequence of numbers'
            degrees = [degrees]
            
        self.degrees = degrees
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, img):
        degree = random.choice(self.degrees)
        return F.rotate(img, degree, self.interpolation, self.expand, self.center, self.fill)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', interpolation={0}'.format(self.interpolation)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomGaussBlur(object):
    """Gaussian Blurring on image by random radius parameter.
    Args:
        radius (number or sequence): radius for gaussian blurring (also known as sigma)
            if sequence (radius_min, radius_max): the radius is randomly sampled from this range
            recommended to be < 2
    """
    def __init__(self, sigma):
        _log_api_usage_once(self)
        if isinstance(sigma, Sequence):
            assert isinstance(sigma[0], Number) and isinstance(sigma[1], Number), \
                "elements of radius should be numbers."
        else:
            assert isinstance(sigma, Number), \
                "radius should be a single number or a range of (radius_min, radius_max)."
            sigma = [sigma, sigma]
        
        self.sigma = sigma

    def __call__(self, img):
        sigma = torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()

        ks = round(sigma*6 + 1)|1 # kernel size calculation of OpenCV: ksize.width = cvRound(sigma1*(depth == CV_8U ? 3 : 4)*2 + 1)|1

        return F.gaussian_blur(img, (ks,ks), sigma)

    def __repr__(self):
        return self.__class__.__name__ + '(Gaussian Blur radius={0})'.format(self.radius)

class RandomGaussNoise(object):
    """Additive Gaussian Noise on image by random sigma parameter.
    Args:
        sigma (number or sequence): sigma for gaussian noise
            if sequence (sigma_min, sigma_max): sigma is randomly sampled from this range
    """
    def __init__(self, sigma):
        _log_api_usage_once(self)
        if isinstance(sigma, Sequence):
            assert isinstance(sigma[0], Number) and isinstance(sigma[1], Number), \
                "elements of sigma should be numbers."
        else:
            assert isinstance(sigma, Number), \
                "sigma should be a single number or a range of (sigma_min, sigma_max)."
            sigma = [sigma, sigma]
        
        self.sigma = sigma

    def __call__(self, img):
        sigma = torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()

        if isinstance(img, torch.Tensor):
            img = img + img.clone().normal_(std=sigma) #torch.normal(0, sigma, img.shape).device(img.device) #img.normal_(std=sigma)#, img.shape)
            return img
        else:
            img = np.array(img)
            noise = np.random.normal(0,sigma*255,img.shape)
            img = (img+noise).clip(0,255).astype('uint8')
            return Image.fromarray(img)

    def __repr__(self):
        return self.__class__.__name__ + '(Additive Gaussian Noise sigma={0})'.format(self.sigma)
        
class RandomResize(object):
    """Random Resize transformation by scale parameter.
    Args:
        scale (number or sequence): scale factor for resizing
            if sequence (scale_min, scale_max): scale is randomly sampled from this range
    """
    def __init__(
        self, scale,
        interpolation=InterpolationMode.BILINEAR,
        max_size=None,
        antialias=None
    ):
        _log_api_usage_once(self)
        if isinstance(scale, Sequence):
            assert isinstance(scale[0], Number) and isinstance(scale[1], Number), \
                "elements of scale should be numbers."
        else:
            assert isinstance(scale, Number), \
                "scale should be a single number or a range of (scale_min, scale_max)."
            scale = [scale, scale]
        
        self.scale = scale
        self.interpolation=interpolation
        self.max_size=max_size
        self.antialias=antialias

    def __call__(self, img):
        scale = torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
        _, height, width = F.get_dimensions(img)
        size = (round(scale*height), round(scale*width))
        img = F.resize(img, size, self.interpolation, self.max_size, self.antialias)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(scale value={0})'.format(self.scale)

class RandomAffineCV2(object):
    """Random Affine transformation by CV2 method on image by alpha parameter.
    Args:
        alpha (float): alpha value for affine transformation
        mask (PIL Image) in __call__, if not assign, set None.
    """
    def __init__(self, alpha):
        _log_api_usage_once(self)
        assert isinstance(alpha, Number), "alpha should be a single number."
        assert 0. <= alpha <= 0.15, \
            "In pathological image, alpha should be in (0,0.15), you can change in myTransform.py"
        self.alpha = alpha

    @staticmethod
    def affineTransformCV2(img, alpha, mask=None):
        alpha = img.shape[1] * alpha
        if mask is not None:
            mask = np.array(mask).astype(np.uint8)
            img = np.concatenate((img, mask[..., None]), axis=2)

        imgsize = img.shape[:2]
        center = np.float32(imgsize) // 2
        censize = min(imgsize) // 3
        pts1 = np.float32([center+censize, [center[0]+censize, center[1]-censize], center-censize])  # raw point
        pts2 = pts1 + np.random.uniform(-alpha, alpha, size=pts1.shape).astype(np.float32)  # output point
        M = cv2.getAffineTransform(pts1, pts2)  # affine matrix
        img = cv2.warpAffine(img, M, imgsize[::-1],
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        if mask is not None:
            return Image.fromarray(img[..., :3]), Image.fromarray(img[..., 3])
        else:
            return Image.fromarray(img)

    def __call__(self, img, mask=None):
        return self.affineTransformCV2(np.array(img), self.alpha, mask)

    def __repr__(self):
        return self.__class__.__name__ + '(alpha value={0})'.format(self.alpha)


class RandomElastic(object):
    """Elastic distotion of images using alpha, sigma parameter as described in [Simard2003]_.
    Based on https://gist.github.com/ernestum/601cdf56d2b424757de5
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    Args:
        alpha (float, int): alpha value for Elastic transformation
            if alpha is 0, output is original whatever the sigma;
            if float:       factor relavtiv to image width
                if alpha is 1.0, output only depends on sigma parameter;
                if alpha < 1.0 or > 1.0, it zoom in or out the sigma's Relevant dx, dy.
            if int:         absolute value
            if sequence:    range of (alpha_min, alpha_max)
        sigma (float): sigma value for Elastic transformation, should be \ in (0.05,0.1)
            if float:       factor relavtiv to image width
            if int:         absolute value
            if sequence:    range of (sigma_min, sigma_max)
        mask (PIL Image) in __call__, if not assign, set None.
        interpolation: Default: PIL.Image.BILINEAR
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is reflect.
    """
    def __init__(self, alpha, sigma, interpolation=Image.BILINEAR, padding_mode='reflect'):
        _log_api_usage_once(self)
        if isinstance(alpha, Sequence):
            assert isinstance(alpha[0], Number) and isinstance(alpha[1], Number), \
                "elements of alpha should be numbers."
        else:
            assert isinstance(alpha, Number), \
                "alpha should be a single number or a range of (alpha_min, alpha_max)."
            alpha = [alpha, alpha]
        
        if isinstance(sigma, Sequence):
            assert isinstance(sigma[0], Number) and isinstance(sigma[1], Number), \
                "elements of sigma should be numbers."
        else:
            assert isinstance(sigma, Number), \
                "sigma should be a single number or a range of (sigma_min, sigma_max)."
            sigma = [sigma, sigma]

        for s in sigma:
            if isinstance(s, float):
                assert 0.05 <= s <= 0.1, \
                    "In pathological image, sigma should be in (0.05,0.1)"

        self.alpha = alpha
        self.sigma = sigma
        self.interpolation = interpolation
        self.padding_mode = padding_mode

    @staticmethod
    def RandomElasticCV2(img, alpha, sigma, mask=None, interpolation=Image.BILINEAR, padding_mode='reflect'):
        
        _, height, width = F.get_dimensions(img)

        for i in range(2):
            if isinstance(alpha[i], float):
                alpha[i] = alpha[i] * max(height, width)
            if isinstance(sigma[i], float):
                sigma[i] = sigma[i] * max(height, width)

        alpha = torch.empty(1).uniform_(alpha[0], alpha[1]).item()
        sigma = torch.empty(1).uniform_(sigma[0], sigma[1]).item()
        if mask is not None:
            img = torch.stack((img, torch.tensor(mask)), 1)

        ks = round(sigma*6 + 1)|1 # kernel size calculation of OpenCV: ksize.width = cvRound(sigma1*(depth == CV_8U ? 3 : 4)*2 + 1)|1

        dgrid = F.gaussian_blur(torch.rand((2, height, width), device=img.device) * 2 - 1, (ks,ks), sigma) #torch.nn.functional.normalize(, p=2, dim=0)
        grid = torch.stack(torch.meshgrid(torch.arange(width, device=img.device), torch.arange(height, device=img.device), indexing='xy')) + dgrid * alpha
        y, x = grid[0]/height * 2 - 1, grid[1]/width * 2 - 1

        grid = torch.stack((y, x), 2).expand(img.shape[0],-1,-1,-1).to(img.device)
        img = torch.nn.functional.grid_sample(img, grid, padding_mode="reflection", align_corners=False)

        if mask is not None:
            return img[:, :3, ...], img[:, 3, ...]
        else:
            return img

    def __call__(self, img, mask=None):
        return self.RandomElasticCV2(img, self.alpha, self.sigma, mask, self.interpolation, self.padding_mode) #np.array(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(alpha value={0})'.format(self.alpha)
        format_string += ', sigma={0}'.format(self.sigma)
        format_string += ')'
        return format_string
