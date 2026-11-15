import json
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace
from feature_pipeline import count_kmers, fasta_iter, parse_taxonomy, label_from_taxonomy, text_records, valid_record, Record

class FormatTests(unittest.TestCase):
    def test_dna_stride_one_repeats(self):
        counts = count_kmers('AAAAAAAAA', 8)
        self.assertEqual(sum(counts.values()), 2)
        self.assertEqual(len(counts), 1)
    def test_no_bridge_across_n(self):
        self.assertEqual(sum(count_kmers('AAAAAAAANAAAAAAAA', 8).values()), 2)
        self.assertFalse(valid_record(Record('id', 'x', 'AAAAANAAAA'), Namespace(kind='dna', k=8)))
    def test_multiline_fasta(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'a.fa'
            p.write_text('>seq1\nACGT\nACGT\n>seq2\nAAAA\n')
            self.assertEqual(list(fasta_iter(p)), [('seq1', 'ACGTACGT'), ('seq2', 'AAAA')])
    def test_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'docs.jsonl'
            p.write_text(json.dumps({'doc_id':'q1','topic':'sports','body':'a b c'})+'\n')
            args = Namespace(input=p,text_format='jsonl',label_col='topic',text_col='body',id_col='doc_id')
            a=list(text_records(args))
            self.assertEqual([(r.identifier,r.label,r.content) for r in a],[('q1','sports','a b c')])
    def test_genus_path_uses_parent_taxon(self):
        tax=parse_taxonomy('Bacteria;A;B;C;D;E', 'six', 'Bacteria')
        self.assertEqual(label_from_taxonomy(tax,'family'), 'Bacteria;A;B;C;D')
        self.assertEqual(label_from_taxonomy(tax,'genus'), 'Bacteria;A;B;C;D;E')

if __name__=='__main__': unittest.main()