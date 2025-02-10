import argparse, os
from videsrgan.realesrgan import RealESRGAN
from videsrgan.realesrganfx import RealESRGANFx
from moviepy import VideoFileClip, vfx

def int_constraint(input, constraint):
    try:      
        value = int(input)
        if not constraint(value):
            raise ValueError()
    except:
        raise argparse.ArgumentTypeError(f"'{input}' is not a valid value")
    return value

def tilesize_type(input):
    return int_constraint(input, (lambda x : (x == 0 or x >= 32)))

def dir_type(input):
    return input
    if os.path.isdir(input):
        return input
    else:
        raise argparse.ArgumentTypeError(f"'{input}' is not a valid path")
    
def gpuid_type(input):
    return int_constraint(input, (lambda x : (x >= -1)))

def scale_type(input):
    scales = (2,3,4)
    value = int(input)
    if value in scales:
        return value
    else:
        raise argparse.ArgumentTypeError(f"'{input}' is not a valid scale")
    
def crf_type(input):
    return int_constraint(input, (lambda x : (x >= 0 and x <= 51)))

def main():  # pragma: no cover   
    parser = argparse.ArgumentParser(prog='videnh', description='What the program does')

    parser.add_argument('-i', help='input video path', type=argparse.FileType('r', encoding='UTF-8'), required=True)
    parser.add_argument('-o', help='output video path', type=argparse.FileType('w', encoding='UTF-8'))
    parser.add_argument('-t', help='tile size (>=32/0=auto, default=0)', type=tilesize_type, default=0)
    parser.add_argument('-m', help='folder path to the pre-trained models. default=models', type=dir_type, default='./models')
    parser.add_argument('-n', help='model name (default=RealESRGAN_General_WDN_x4_v3, can be RealESRGAN_General_WDN_x4_v3 | realesrgan-x4plus)', type=str, default='RealESRGAN_General_WDN_x4_v3')
    parser.add_argument('-g', help='gpu device to use (default=auto) can be 0,1,2 for multi-gpu', type=gpuid_type, default=-1)
    parser.add_argument('-s', help='upscale ratio (can be 2, 3, 4. default=4)', type=scale_type, default=4)
    parser.add_argument('-q', help='a lower CRF gives better video quality (0>=51, default=17)', type=crf_type, default=17)
    parser.add_argument('-x', help='enable tta mode', action='store_true')
    parser.add_argument('-v', help='verbose output', action='store_true')

    args = parser.parse_args()

    realesrgan = RealESRGAN(
        gpuid=args.g, 
        tta_mode=args.x, 
        tilesize=args.t, 
        scale=args.s,
        model_path=args.m,
        model=args.n
    )

    input_video = VideoFileClip(args.i.name)

    rotated_video = input_video.with_effects([RealESRGANFx(realesrgan)])

    rotated_video.write_videofile(args.o.name, codec='libx264', ffmpeg_params=["-crf", str(args.q)])


