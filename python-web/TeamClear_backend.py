from __future__ import absolute_import
from __future__ import print_function

import cv2
import argparse
import numpy as np
import tensorflow as tf


def decode_video(filename, n_frames=0, fps=-1, random_chunk=False):
    """
    Decode frames from a video. Returns frames in [0, 255] RGB format.
    :param filename: string tensor, e.g. dequeue() op from a filenames queue
    :return:
        video: 4-D tensor containing frames of a video: [time, height, width, channel]
        height: frame height
        width: frame width
        length: number of non-zero frames loaded from the video (the rest of the sequence is zero-padded)
    """
    params = [filename, n_frames, fps, random_chunk]
    dtypes = [tf.float32, tf.int64, tf.int64, tf.int64]
    return tf.py_func(_load_video_ffmpeg, params, dtypes, name='decode_video')


def _parse_arguments():
    parser = argparse.ArgumentParser('Test Endecoder python.')
    parser.add_argument('-i', '--input',
          help='Please insert the video file path.')
    parser.add_argument('-o', '--output', default=None, 
    	    help='(Optional) Path to the .npy file where the decoded frames will be stored.')
    parser.add_argument('-p', '--play', default=False, action='store_true',
          help='Play the extracted frames.')
    parser.add_argument('-s', '--speed', default=-1, type=int, 
    	    help='Framerate to which the input videos are converted. Use -1 for the original framerate.')
    return parser.parse_args()


def _show_video(video, fps):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pylab as pl
    pl.style.use('ggplot')
    pl.axis('off')

    if fps < 0:
        fps = 25
    video /= 255.  # Pylab works in [0, 1] range
    img = None
    pause_length = 1. / fps
    try:
        for f in range(video.shape[0]):
            im = video[f, :, :, :]
            if img is None:
                img = pl.imshow(im)
            else:
                img.set_data(im)
            pl.pause(pause_length)
            pl.draw()
    except:
        pass


if __name__ == '__main__':
    args = _parse_arguments()
    sess = tf.Session()
    f = tf.placeholder(tf.string)
    video, h, w, seq_length = decode_video(f, args.num_frames, args.fps, args.random_chunks)
    start_time = time.time()
    frames, seq_length_val = sess.run([video, seq_length], feed_dict={f: args.input_file})
    total_time = time.time() - start_time
    print('\nSuccessfully loaded video!\n'
          '\tDimensions: %s\n'
          '\tTime: %.3fs\n'
          '\tLoaded frames: %d\n' % (str(frames.shape), total_time, seq_length_val))
    if args.output_file:
        np.save(args.output_file, frames)
        print("Stored frames to %s" % args.output_file)
    if args.play_video:
        _show_video(frames, args.fps)

