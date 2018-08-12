import errno
import os
import re
import shutil
import subprocess
import sys


def natural_sort(unsorted_list):
    """Used to sort lists like a human
    e.g. [1, 10, 11, 2, 3] to [1, 2, 3, 10, 11]
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(unsorted_list, key=alphanum_key)


def copy_and_rename_plots(plot_lattice_dir, output_dir):
    """
    Given a plot_lattice or plot_grid directory, copies it and renames the files in an order that
    ffmpeg.exe likes, e.g. (0001, 0002, 0003, ... , 0010, 0011, ...)
    Notes:
        - assumes less than 10000 files are being copied (for ffmpeg simplicity)
    """
    # copy the folder
    try:
        shutil.copytree(plot_lattice_dir, output_dir)
    except OSError as exc:
        if exc.errno == errno.ENOTDIR:
            shutil.copy(plot_lattice_dir, output_dir)
        else:
            raise
    # naturally sort the copied files
    unsorted_files = os.listdir(plot_lattice_dir)
    assert len(unsorted_files) <= 99999  # assume <= 5 digits later
    sorted_files = natural_sort(unsorted_files)
    # rename the files accordingly
    basename = "lattice_at_time_"
    filetype = ".jpg"
    for i, filename in enumerate(sorted_files):
        num = "%05d" % i
        newname = basename + num + filetype
        os.rename(os.path.join(output_dir, filename), os.path.join(output_dir, newname))
    return


def make_video_ffmpeg(plot_lattice_dir, output_path, fps=5, ffmpeg_dir=None):
    """Makes a video using ffmpeg - also copies the lattice plot dir, changes filenames, and deletes the copy
    Args:
        plot_lattice_dir: source directory
        output_path: path and filename of the output video
        fps: frames per second of the output video
        ffmpeg_dir: [default: None] location of the ffmpeg directory (root where bin containing ffmpeg.exe is)
    Returns:
        None
    Notes:
        - assumes ffmpeg has been extracted on your system and added to the path
        - if it's not added to path, point to it (the directory containing ffmpeg bin) using ffmpeg_dir arg
        - assumes less than 10000 images are being joined (for ffmpeg simplicity)
        - .mp4 seems to play best with Windows Media Player, not VLC
    """
    # make sure video directory exists
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    # make temp dir
    temp_plot_dir = os.path.join(plot_lattice_dir, os.pardir, "temp")
    copy_and_rename_plots(plot_lattice_dir, temp_plot_dir)
    # make video
    command_line = ["ffmpeg",
                    "-framerate", "%d" % fps,                                       # *force* video frames per second
                    "-i", os.path.join(temp_plot_dir, "lattice_at_time_%05d.jpg"),  # set the input files
                    "-vcodec", "libx264",                                           # set the video codec
                    "-r", "%d" % fps,                                               # set video frames per second
                    "-pix_fmt", "yuv420p",                                          # pixel formatting
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",                     # fix if height/width not even ints
                    "%s" % output_path]                                             # output path of video
    if ffmpeg_dir is not None:
        app_path = os.path.join(ffmpeg_dir, "bin", "ffmpeg.exe")
        sp = subprocess.Popen(command_line, executable=app_path, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    else:
        sp = subprocess.Popen(command_line, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        while True:
            out = sp.stderr.read(1)
            if out == '' and sp.poll() != None:
                break
            if out != '':
                sys.stdout.write(out)
                sys.stdout.flush()
    out, err = sp.communicate()
    print out, err, sp.returncode
    # delete temp dir
    shutil.rmtree(temp_plot_dir)
    return
