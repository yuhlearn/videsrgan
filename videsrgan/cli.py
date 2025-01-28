import argparse, os, av
import av.audio
import av.codec

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
    return int_constraint(input, (lambda x : (x >= 0)))

def crf_type(input):
    return int_constraint(input, (lambda x : (x >= 0 and x <= 51)))

def threadcount_type(input):
    threadcount = list()

    for thread in input.split(':'):
        value = int_constraint(thread, (lambda x : (x > 0)))
        threadcount.append(value)

    if len(threadcount) != 3:
        raise argparse.ArgumentTypeError(f"'{input}' is not a valid value")

    return tuple(threadcount)

def main():  # pragma: no cover
    parser = argparse.ArgumentParser(prog='videnh', description='What the program does')

    parser.add_argument('-i', help='input video path', type=argparse.FileType('r', encoding='UTF-8'), required=True)
    parser.add_argument('-o', help='output video path', type=argparse.FileType('w', encoding='UTF-8'))
    parser.add_argument('-t', help='tile size (>=32/0=auto, default=0)', type=tilesize_type, default=0)
    parser.add_argument('-m', help='folder path to the pre-trained models. default=models', type=dir_type, default='models')
    parser.add_argument('-n', help='model name (default=realesr-animevideov3, can be realesr-animevideov3 | realesrgan-x4plus | realesrgan-x4plus-anime | realesrnet-x4plus)', default='realesr-animevideov3')
    parser.add_argument('-g', help='gpu device to use (default=auto) can be 0,1,2 for multi-gpu', type=gpuid_type)
    parser.add_argument('-j', help='thread count for load/proc/save (default=1:2:2)', type=threadcount_type, default=(1,2,2))
    parser.add_argument('-q', help='Constant Rate Factor -- a lower CRF gives better video quality (0>=51, default=17)', type=crf_type, default=17)
    parser.add_argument('-x', help='enable tta mode', action='store_true')
    parser.add_argument('-v', help='verbose output', action='store_true')

    args = parser.parse_args()

    model_path = '/usr/local/realesrgan-ncnn-vulkan-20220424-ubuntu/models/RealESRGAN_General_x4_v3'

    with av.open(args.o.name, 'w') as output_container:
        with av.open(args.i.name, 'r') as input_container:
            print('Finished video encoding.')

            # # Begin work on the audio stream(s)
            # for input_audio_stream in input_container.streams.audio:
            #     output_audio_stream = output_container.add_stream('mp3')
            #     output_audio_stream.layout = 'stereo'

            #     for input_audio_frame in input_container.decode(input_audio_stream):
            #         for output_audio_packet in output_audio_stream.encode(input_audio_frame):
            #             output_container.mux(output_audio_packet)
                
            #     # Flush the audio stream
            #     output_container.mux(output_audio_stream.encode())

            #     print('Finished audio encoding.')

            # Begin work on the video stream
            input_video_stream = input_container.streams.video[0]

            output_video_stream = output_container.add_stream('h264', input_video_stream.guessed_rate)
            output_video_stream.pix_fmt = 'yuv420p'
            output_video_stream.width = input_video_stream.width
            output_video_stream.height = input_video_stream.height
            output_video_stream.options = {
                'crf' : str(args.q), 
                'preset' : 'veryslow', 
                'tune' : 'film'
            }
            
            for input_video_frame in input_container.decode(input_video_stream):
                input_video_image = input_video_frame.to_image()

                # Do someting to the image
                output_video_image = input_video_image

                output_video_frame = av.VideoFrame.from_image(output_video_image)
                output_video_frame.pts = None

                for output_video_packet in output_video_stream.encode(output_video_frame):
                    output_container.mux(output_video_packet)

            # Flush the video stream
            output_container.mux(output_video_stream.encode())