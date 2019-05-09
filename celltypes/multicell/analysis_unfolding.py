import numpy as np
import os

from singlecell.singlecell_simsetup import singlecell_simsetup
from singlecell.singlecell_constants import RUNS_FOLDER
from utils.make_video import make_video_ffmpeg


def scan_gamma_dynamics(J, W, state, coordnum=8, verbose=False):
    critgamma = None
    for gamma in np.linspace(0.001, 0.1, 10000):
        Js_internal = np.dot(J, state)
        state_01 = (state + np.ones_like(state)) / 2.0
        h_field_nbr = gamma * coordnum * np.dot(W, state_01)
        updated_state = np.sign(Js_internal + h_field_nbr)
        if np.array_equal(updated_state, state):
            if verbose:
                print gamma, True
        else:
            if critgamma is None:
                critgamma = gamma
            if verbose:
                print gamma, False
    return critgamma


if __name__ == '__main__':
    make_single_video = False
    batch_make_videos = False  # make videos in all runs subdirs
    destabilize_celltypes_gamma = True

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
        overlapref = False
        basedir = RUNS_FOLDER + os.sep + "Annotated Multicell Data  Feb 2019" + os.sep + "other2"
        if overlapref:
            flagmod = " (overlapRef)"
            source_dir = "lattice" + os.sep + "overlapRef_0_0"
            fhead = "lattice_overlapRef_0_0"
        else:
            flagmod = ""
            source_dir = "lattice"
            fhead = "composite_lattice_step"
        ftype = ".png"
        nmax = 20
        fps = 1
        dirnames = os.listdir(basedir)
        rundirs = [os.path.join(basedir, o) for o in dirnames if os.path.isdir(os.path.join(basedir, o))]
        for idx, run in enumerate(rundirs):
            sourcepath = run + os.sep + source_dir
            outpath = run + os.sep + '%s%s.mp4' % (dirnames[idx], flagmod)
            make_video_ffmpeg(sourcepath, outpath, fps=1, fhead=fhead, ftype=ftype, nmax=nmax)

    if destabilize_celltypes_gamma:
        random_mem = False
        random_W = False
        simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W)
        coordnum = 8  # num neighbours which signals are received from
        W = simsetup['FIELD_SEND']
        J = simsetup['J']
        celltypes = (simsetup['XI'][:, a] for a in xrange(simsetup['P']))
        print 'Scanning for monotype destabilizing gamma (for coordination number %d)' % coordnum
        for idx, celltype in enumerate(celltypes):
            critgamma = scan_gamma_dynamics(J, W, celltype, coordnum=coordnum, verbose=False)
            print idx, simsetup['CELLTYPE_LABELS'][idx], critgamma