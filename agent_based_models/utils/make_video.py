import errno
import os
import re
import shutil
import subprocess


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
    assert len(unsorted_files) <= 9999  # assume <= 4 digits later
    sorted_files = natural_sort(unsorted_files)
    # rename the files accordingly
    basename = "lattice_at_time_"
    filetype = ".png"
    for i, filename in enumerate(sorted_files):
        num = "%04d" % i
        newname = basename + num + filetype
        os.rename(os.path.join(output_dir, filename), os.path.join(output_dir, newname))
    return


def make_video_ffmpeg(plot_lattice_dir, ffmpeg_dir, output_path, fps):
    """Makes a video using ffmpeg - also copies the lattice plot dir, changes filenames, and deletes the copy
    Args:
        plot_lattice_dir: source directory
        ffmpeg_dir: location of the ffmpeg directory (root where bin containing ffmpeg.exe is)
        output_path: path and filename of the output video
        fps: frames per second of the output video
    Returns:
        None
    """
    # make temp dir
    temp_plot_dir = os.path.join(ffmpeg_dir, "bin", "temp")
    copy_and_rename_plots(plot_lattice_dir, temp_plot_dir)

    # make video
    app_path = os.path.join(ffmpeg_dir, "bin", "ffmpeg.exe")
    command_line = ["ffmpeg", "-framerate", "%d" % fps, "-i", os.path.join(temp_plot_dir, "lattice_at_time_%04d.png"),
                    "-c:v", "libx264", "-r", "%d" % fps, "-pix_fmt", "yuv420p", "%s" % output_path]

    sp = subprocess.Popen(command_line, executable=app_path, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = sp.communicate()
    print out, err, sp.returncode

    # delete temp dir
    shutil.rmtree(temp_plot_dir)

    return
