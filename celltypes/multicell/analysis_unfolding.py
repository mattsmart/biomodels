import os

from singlecell.singlecell_constants import RUNS_FOLDER
from utils.make_video import make_video_ffmpeg


if __name__ == '__main__':
    make_single_video = True
    batch_make_videos = False  # make videos in all runs subdirs

    if make_single_video:
        basedir = RUNS_FOLDER + os.sep + "expC1_fsHigh_beta1.0_radius1"
        source_dir = "lattice" + os.sep + "overlapRef_0_0"
        #fhead = "composite_lattice_step"
        fhead = "lattice_overlapRef_0_0_step"
        ftype = ".png"
        nmax = 100
        fps = 2
        sourcepath = basedir + os.sep + source_dir
        outpath = basedir + os.sep + 'movie2_expC1_fsHigh_beta1.mp4'
        make_video_ffmpeg(sourcepath, outpath, fps=1, fhead=fhead, ftype=ftype, nmax=nmax)

    if batch_make_videos:
        basedir = RUNS_FOLDER + os.sep + "expC1_fsHigh_beta1.0_radius1"
        source_dir = "lattice"
        fhead = "composite_lattice_step"
        ftype = ".png"
        nmax = 20
        fps = 1
        rundirs = [os.path.join(basedir, o) for o in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, o))]
        for idx, run in enumerate(rundirs):
            sourcepath = run + os.sep + source_dir
            outpath = run + os.sep + 'random%d.mp4' % idx
            make_video_ffmpeg(sourcepath, outpath, fps=1, fhead=fhead, ftype=ftype, nmax=nmax)
