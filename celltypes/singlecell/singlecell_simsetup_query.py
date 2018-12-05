import csv
import mygene
import os

from singlecell_constants import DATADIR
from singlecell_simsetup import singlecell_simsetup

"""
use mygene package https://pypi.org/project/mygene/
allows for synonym search in gene names to answer questions such as:
'Does the single cell dataset contain targets of miR-21?'
"""


TAXID_MOUSE = 10090


def print_simsetup_labels(simsetup):
    print 'Genes:'
    for idx, label in enumerate(simsetup['GENE_LABELS']):
        print idx, label
    print 'Celltypes:'
    for idx, label in enumerate(simsetup['CELLTYPE_LABELS']):
        print idx, label


def read_gene_list_csv(csvpath, aliases=False):
    with open(csvpath, 'r') as f:
        reader = csv.reader(f)
        if aliases:
            data = {r[0]: r[1:] for r in reader}
        else:
            data = [r[0] for r in reader]
    return data


def get_mygene_hits(gene_symbol, taxid=TAXID_MOUSE):
    mg = mygene.MyGeneInfo()
    query = mg.query(gene_symbol)
    # print query
    # print query.keys()
    hits = []
    for hit in query['hits']:
        if hit['taxid'] == taxid:
            #print hit
            hits.append(hit)
    return hits


def collect_mygene_hits(gene_symbol_list, taxid=TAXID_MOUSE, entrez_compress=True):
    """
    Returns: two dicts: gene_hits, hitcounts
    gene_hits is dictionary of form:
        {gene_symbol:
                [{hit_1}, {hit_2}, ..., {hit_N}]}
    if entrez_compress:
        {gene_symbol:
                [entrez_id_1, ..., entrez_id_P]} where P < N is number of unique entrez ids
    hitcounts is of the form
    """
    gene_hits = {}
    hitcounts = {}

    # if gene_list arg is path, load the list it contains
    if isinstance(gene_symbol_list, basestring):
        if os.path.exists(gene_symbol_list):
            print "loading data from %s" % gene_symbol_list
            gene_symbol_list = read_gene_list_csv(gene_symbol_list)

    print "Searching through %d genes..." % len(gene_symbol_list)
    for idx, g in enumerate(gene_symbol_list):
        hits = get_mygene_hits(g, taxid=taxid)
        if entrez_compress:
            hits = [int(h.get('entrezgene', 0)) for h in hits]
            hits = [h for h in hits if h > 0]
            if len(hits) > 1:
                print "WARNING len(hits)=%d > 1 for %s (%d)" % (len(hits), g, idx)
        gene_hits[g] = hits

        if len(hits) in hitcounts.keys():
            hitcounts[len(hits)][g] = hits
        else:
            hitcounts[len(hits)] = {g: hits}
        if idx % 100 == 0:
            print "Progress: %d of %d" % (idx, len(gene_symbol_list))

    # print some stats
    for count in xrange(max(hitcounts.keys())+1):
        if count in hitcounts.keys():
            print "Found %d with %d hits" % (len(hitcounts[count].keys()), count)
        else:
            print "Found 0 with %d hits" % count
    return gene_hits, hitcounts


def write_genelist_id_csv(gene_list, gene_hits, outpath='genelist_id.csv'):
    with open(outpath, 'w') as fcsv:
        for idx, gene in enumerate(gene_list):
            info_list = [gene] + gene_hits[gene]
            fcsv.write(','.join(str(s) for s in info_list) + '\n')
    return outpath


if __name__ == '__main__':
    simsetup = singlecell_simsetup()
    # print_simsetup_labels(simsetup)

    # find ref for gene list# load csv to compare gene list
    data_genes = simsetup['GENE_LABELS']
    data_genes_lowercase = [g.lower() for g in data_genes]

    """    
    data_gene_hits, data_hitcounts = collect_mygene_hits(data_genes)
    write_genelist_id_csv(data_genes, data_gene_hits)
    """

    hits = get_mygene_hits('MLL3')
    for hit in hits:
        print hit.keys(), hit['taxid']
        print hit

    # check weird hits individually
    """
    for g in data_hitdict[2].keys():
        entrez_ids = []
        for hit in data_hitdict[2][g]:
            hit.get()
            entrez_ids.append('entrezgene', None)
        data_hitdict[2][g] = entrez_ids
        print g
        print data_hitdict[2]
        hits = get_mygene_hits(g)
        print hits
    """

    # # load csv to compare gene list to target database
    targetdir = DATADIR + os.sep + 'misc' + os.sep + 'mir21_targets'
    target_names = ['mir21_misc', 'mir21_wiki', 'mir21_targetscan']
    target_dict = {name: {'path': targetdir + os.sep + '%s.csv' % name} for name in target_names}
    for name in target_names:
        genes = read_gene_list_csv(target_dict[name]['path'])
        print genes
        target_dict[name]['genes'] = genes
        gene_hits, hitcounts = collect_mygene_hits(genes)
        write_genelist_id_csv(genes, gene_hits, outpath='genelist_id_%s.csv' % name)

    """
    with open('targetscan_mir21_barelist.csv', 'r') as targetfile:
        for idx, gene in enumerate(targetfile):
            gene_lowercase = gene.lower()
            if gene_lowercase in reference_list:
                print "Target", idx, gene_lowercase
    """