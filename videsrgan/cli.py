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
    if os.path.isdir(input):
        return input
    else:
        raise argparse.ArgumentTypeError(f"'{input}' is not a valid path")
    
def gpuid_type(input):
    return int_constraint(input, (lambda x : (x >= -1)))

def threads_type(input):
    return int_constraint(input, (lambda x : (x >= 1)))

def scale_type(input):
    scales = (2,3,4)
    value = int(input)
    if value in scales:
        return value
    else:
        raise argparse.ArgumentTypeError(f"'{input}' is not a valid scale")
    
def param_type(input):
    return input

def main():  # pragma: no cover
    usage='videsrgan -i INFILE -o OUTFILE [OPTIONS] [--params [FFMPEG OPTIONS]]'
    description='Upscale videos using Real ESRGAN NCNN Vulkan and FFMPEG.'
    epilog=''

    parser = argparse.ArgumentParser(
        prog='videsrgan',
        usage=usage,
        description=description,
        epilog=epilog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    
    esrgan_group = parser.add_argument_group('Real ESRGAN options')
    ffmpeg_group = parser.add_argument_group('FFMPEG options')

    # Basic options
    parser.add_argument(
        '-i', '--input',
        help='input video path',
        dest='input',
        metavar='F',
        type=argparse.FileType('r', encoding='UTF-8'), 
        required=True
        )
    parser.add_argument(
        '-o', '--output',
        help='output video path', 
        dest='output',
        metavar='F',
        type=argparse.FileType('w', encoding='UTF-8'),
        required=True
        )
    
    # ESRGAN options
    esrgan_group.add_argument(
        '-g', '--gpu',
        help='gpu device to use (>=0 | -1=auto)',
        dest='gpu',
        metavar='I',
        type=gpuid_type, 
        default=-1
        )
    esrgan_group.add_argument(
        '-m', '--model_dir',
        help='folder path to the pre-trained models',
        dest='model_dir',
        metavar='D',
        type=dir_type, 
        default=os.path.dirname(__file__) + '/models/'
        )
    esrgan_group.add_argument(
        '-n', '--model',
        help='model name (RealESRGAN_General_WDN_x4_v3 | realesrgan-x4plus)', 
        dest='model',
        metavar='S',
        type=str, 
        default='RealESRGAN_General_WDN_x4_v3'
        )
    esrgan_group.add_argument(
        '-s', '--scale',
        help='upscale ratio (2 | 3 | 4) must match the model, which is usually indicated by the name e.g. "4x"',
        dest='scale',
        metavar='I',
        type=scale_type, 
        default=4
        )
    esrgan_group.add_argument(
        '-t', '--tile_size',
        help='tile size (>=32 | 0=auto)',
        dest='tile_size',
        metavar='I',
        type=tilesize_type, 
        default=0
        )
    esrgan_group.add_argument(
        '-x', '--tta',
        help='enable tta mode',
        dest='tta',
        action='store_true'
        )
    
    # FFMPEG options
    ffmpeg_group.add_argument(
        '-c', '--codec',
        help='codec (libx264 | rawvideo | png | libvpx or any other codec supported by ffmpeg)',
        dest='codec',
        metavar='S',
        type=str, 
        default='libx264'
        )
    ffmpeg_group.add_argument(
        '-f', '--pix_fmt',
        help='pixel format for the output video file (run "ffmpeg -pix_fmts" for available formats)',
        dest='pixel_format',
        metavar='S',
        type=str, 
        default=None
        )
    ffmpeg_group.add_argument(
        '-l', '--log',
        help='write log files for the audio and the video',
        dest='log',
        action='store_true'
        )
    ffmpeg_group.add_argument(
        '-p', '--preset',
        help='preset (ultrafast | superfast | veryfast | faster | fast | medium | slow | slower | veryslow | placebo)',
        dest='preset',
        metavar='S',
        type=str, 
        default='medium'
        )
    ffmpeg_group.add_argument(
        '-u', '--threads',
        help='number of threads to use for ffmpeg (>=1)',
        dest='threads',
        metavar='I',
        type=threads_type, 
        default=None
        )
    ffmpeg_group.add_argument(
        '-', '--params', 
        help='interprets the remaider as ffmpeg parameters (e.g. "-crf 0" to set libx264 constant rate factor)',
        dest='ffmpeg_params',
        nargs=argparse.REMAINDER
        )

    args = parser.parse_args()

    realesrgan = RealESRGAN(
        gpuid=args.gpu, 
        tta_mode=args.tta, 
        tilesize=args.tile_size, 
        scale=args.scale,
        model_path=args.model_dir,
        model=args.model
        )

    input_video = VideoFileClip(args.input.name)

    enhanced_video = input_video.with_effects([RealESRGANFx(realesrgan)])

    enhanced_video.write_videofile(
        filename=args.output.name,
        codec=args.codec,
        preset=args.preset,
        threads=args.threads,
        ffmpeg_params=args.ffmpeg_params,
        pixel_format=args.pixel_format,
        write_logfile=args.log
        )
    
    enhanced_video.close()
    input_video.close()


