from singlecell.singlecell_simsetup import singlecell_simsetup
from multicell_simulate import mc_sim


if __name__ == '__main__':

    n = 20  # global GRIDSIZE
    steps = 20  # global NUM_LATTICE_STEPS
    buildstring = "dual"  # mono/dual/memory_sequence/
    fieldstring = "no_exo_field"  # on/off/all/no_exo_field, note e.g. 'off' means send info about 'off' genes only
    fieldprune = 0.0  # amount of external field idx to randomly prune from each cell
    ext_field_strength = 0.15                                 # global EXT_FIELD_STRENGTH tunes exosomes AND sent field
    #app_field = construct_app_field_from_genes(IPSC_EXTENDED_GENES_EFFECTS, simsetup['GENE_ID'], num_steps=steps)        # size N x timesteps or None
    app_field = None
    app_field_strength = 0.0  # 100.0 global APP_FIELD_STRENGTH
    plot_period = 1
    state_int = True

    num_runs = 20
    ensemble = 1
    for i in xrange(num_runs):
        random_mem = False
        random_W = True
        simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W)
        print "On run %d (%d total)" % (i, num_runs)
        mc_sim(simsetup, gridsize=n, num_steps=steps, buildstring=buildstring, exosome_string=fieldstring,
               field_remove_ratio=fieldprune, ext_field_strength=ext_field_strength, app_field=app_field,
               app_field_strength=app_field_strength, plot_period=plot_period, state_int=state_int)