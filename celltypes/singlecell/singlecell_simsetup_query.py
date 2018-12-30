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
            data = {r[0]: [val for val in r[1:] if val] for r in reader}
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


def check_target_in_gene_id_dict(memories_genes_id, target_genes_id):
    """
    Returns:
        list of tuples (mem_symbol, target_symbol) if they are aliases
    """
    matches = []
    for target_key, target_val in target_genes_id.iteritems():
        for mem_key, mem_val in memories_genes_id.iteritems():
            #print target_key, target_val, mem_key, mem_val
            for target_id in target_val:
                if target_id in mem_val:
                    matches.append((mem_key, target_key))
    return matches


if __name__ == '__main__':
    write_memories_id = False
    write_targets_id = False
    find_matches = True

    if write_memories_id:
        simsetup = singlecell_simsetup()
        memories_genes = simsetup['GENE_LABELS']
        memories_genes_lowercase = [g.lower() for g in memories_genes]
        memories_genes_id, memories_hitcounts = collect_mygene_hits(memories_genes)
        write_genelist_id_csv(memories_genes, memories_genes_id)
    else:
        memories_genes_id = read_gene_list_csv('2014mehta_genelist_id_filled.csv', aliases=True)

    # prep target csv
    targetgenes_dir = DATADIR + os.sep + 'misc' + os.sep + 'mir21_targets'
    targetgenes_id_dir = '.'
    target_names = ['mir21_misc', 'mir21_wiki', 'mir21_targetscan']
    target_dict = {name: {'gene_path': targetgenes_dir + os.sep + '%s.csv' % name} for name in target_names}

    # write target csv to compare gene list to target database
    for name in target_names:
        genes = read_gene_list_csv(target_dict[name]['gene_path'])
        target_dict[name]['genes'] = genes
        if write_targets_id:
            gene_hits, hitcounts = collect_mygene_hits(genes)
            write_genelist_id_csv(genes, gene_hits, outpath=targetgenes_dir + os.sep + 'genelist_id_%s.csv' % name)
        if find_matches:
            target_genes_id = read_gene_list_csv(targetgenes_id_dir + os.sep + 'genelist_id_%s_filled.csv' % name,
                                                 aliases=True)
            # read target csv to compare gene list to target database
            matches = check_target_in_gene_id_dict(memories_genes_id, target_genes_id)
            target_dict[name]['matches'] = matches
            print "MATCHES for %s" % name
            for idx, match in enumerate(matches):
                print match, memories_genes_id[match[0]], target_genes_id[match[1]]
