import mygene

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


def collect_mygene_hits(gene_symbol_list, taxid=TAXID_MOUSE):
    hitcount = {}
    print "Searching through %d genes..." % len(gene_symbol_list)
    for idx, g in enumerate(gene_symbol_list):
        hits = get_mygene_hits(g, taxid=taxid)
        if len(hits) in hitcount.keys():
            hitcount[len(hits)].append(g)
        else:
            hitcount[len(hits)] = [g]
        if idx % 100 == 0:
            print "Progress: %d of %d" % (idx, len(gene_symbol_list))
    for count in xrange(max(hitcount.keys())+1):
        if count in hitcount.keys():
            print "Found %d with %d hits" % (len(hitcount[count]), count)
        else:
            print "Found 0 with %d hits" % (len(hitcount[count]))


if __name__ == '__main__':
    simsetup = singlecell_simsetup()

    # print info from simsetup
    print_simsetup_labels(simsetup)

    # load csv to compare gene list to target database
    data_genes = simsetup['GENE_LABELS']
    data_genes_lowercase = [g.lower() for g in data_genes]
    hitdict = collect_mygene_hits(data_genes[1100:])

    # sweep through target list
    """
    with open('targetscan_mir21_barelist.csv', 'r') as targetfile:
        for idx, gene in enumerate(targetfile):
            gene_lowercase = gene.lower()
            if gene_lowercase in reference_list:
                print "Target", idx, gene_lowercase
    """