import os

if __name__ == '__main__':

    flag_attach_clusters_resave = False

    if flag_attach_clusters_resave:
        compressed_file = datadir + os.sep + "arr_genes_cells_raw_compressed.npz"
        clusterpath = datadir + os.sep + "SI_cells_to_clusters.csv"
        arr, genes, cells = attach_cluster_id_arr_manual(compressed_file, clusterpath, save=True)