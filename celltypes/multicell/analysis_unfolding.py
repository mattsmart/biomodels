import os

from singlecell.singlecell_constants import RUNS_FOLDER
from utils.make_video import make_video_ffmpeg


if __name__ == '__main__':
    batch_make_videos = True  # make videos in all runs subdirs

    if batch_make_videos:
        basedir = RUNS_FOLDER + os.sep + "weird celltype random W runs"
        source_dir = "lattice"
        fhead = "composite_lattice_step"
        ftype = ".png"
        nmax = 20
        fps = 1
        rundirs = [os.path.join(basedir, o) for o in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, o))]
        for idx, run in enumerate(rundirs):
            sourcepath = run + os.sep + source_dir
            #outpath = run + os.sep + 'random%d.mp4' % idx
            outpath = run + os.sep + 'random13.mp4' % idx
            make_video_ffmpeg(sourcepath, outpath, fps=1, fhead=fhead, ftype=ftype, nmax=nmax)
