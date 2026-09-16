"""Read the published Cedden sequence and endpoint tables without model access."""
from pathlib import Path
import pandas as pd
from docx import Document

def load_original_external(raw_dir):
    """Load the unchanged 40-record original classification panel.

    Original predictions and heterogeneous experimental magnitudes are not
    exposed to the evaluation or training code. Gene identities are recovered
    by exact containment in OGS3, checking both strand orientations.
    """
    import hashlib, json
    root = Path(raw_dir)
    source = root / 'original_external/BeetleFormer_external_test.csv'
    manifest = json.loads((source.parent / 'source_manifest.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['sha256']
    original = pd.read_csv(source)
    assert len(original) == 40 and original.label.value_counts().to_dict() == {1:20, 0:20}
    assert original.sequence.nunique() == 40
    complement = str.maketrans('ACGT', 'TGCA')
    transcripts = []
    header, parts = None, []
    for line in (root / 'ibeetle/OGS3_mRNA.fasta').read_text().splitlines() + ['>END']:
        if line.startswith('>'):
            if header is not None:
                tokens = header.replace('|',' ').split()
                gene = next(t.split('-')[0] for t in tokens if t.startswith('TC'))
                transcripts.append((header, gene, ''.join(parts).upper().replace('U','T')))
            header, parts = line[1:], []
        else:
            parts.append(line.strip())
    rows = []
    for i, r in original.iterrows():
        seq = r.sequence
        assert set(seq) <= set('ACGT') and len(seq) == r.length
        reverse = seq.translate(complement)[::-1]
        hits = []
        for transcript, gene, mrna in transcripts:
            for orientation, query in [('+',seq), ('-',reverse)]:
                start = mrna.find(query)
                if start >= 0:
                    hits.append((transcript, gene, start, start+len(seq), orientation))
        genes = sorted({h[1] for h in hits})
        assert len(genes) == 1, (i, r.gene, genes)
        h = sorted(hits)[0]
        rows.append(dict(id=f'original_external_{i:02d}', original_row=i+2,
            original_gene=r.gene, gene=genes[0], label=int(r.label),
            source_paper=r.source_paper, primary_reference=r.primary_reference,
            sequence=seq, length=len(seq), transcript=h[0], start0=h[2],
            end0=h[3], transcript_orientation=h[4],
            gene_mapping='exact full-fragment containment in OGS3; unique source gene'))
    return pd.DataFrame(rows)

def load_cedden(source_dir):
    src=Path(source_dir)
    genes={'Tc-gawky':'TC006679','Tc-klp61F':'TC008263','Tc-nito':'TC009491','Tc-hr3':'TC008909',
           'Tc-eIF3a':'TC012303','Tc-rpt1':'TC006492','Tc-rpn7':'TC006375','Tc-cyp4g15':'TC008058'}
    primer=pd.read_excel(src/'12915_2025_2219_MOESM1_ESM.xlsx',sheet_name='Primer list 2').fillna('')
    primer=primer[primer.iloc[:,3].eq('F')].copy()
    primer['gene']=primer.iloc[:,0].str.extract(r'(TC\d+)')
    primer['category']=['accessibility' if 'accessibility' in str(v) else
                        ('low_ORF' if 'low' in str(v) else ('high_ORF' if r=='ORF' else 'high_UTR'))
                        for v,r in zip(primer.iloc[:,1],primer.iloc[:,2])]
    rows=[]; counts={}
    for rownum,r in enumerate(Document(src/'12915_2025_2219_MOESM3_ESM.docx').tables[0].rows[1:],start=2):
        name,score,access,region,hazard=[c.text.strip() for c in r.cells]
        gene=genes[name]; i=counts.get(gene,0); counts[gene]=i+1
        categories=['high_ORF','low_ORF','high_UTR'] if name=='Tc-rpn7' else ['high_ORF','low_ORF','accessibility','high_UTR']
        category=categories[i]; p=primer[primer.gene.eq(gene)&primer.category.eq(category)]
        assert len(p)==1
        seq=str(p.iloc[0,5]).upper().replace('U','T')
        assert set(seq)<=set('ACGT') and ('UTR' in str(p.iloc[0,2]))==('UTR' in region)
        rows.append(dict(id=gene+'_'+category,gene=gene,gene_name=name,category=category,region=region,
                         sequence=seq,length=len(seq),dsRIP_score=float(score),accessibility=float(access),
                         hazard_ratio=float(hazard),source_table_row=rownum,
                         source='10.1186/s12915-025-02219-6; Additional file 1 Primer list 2; Additional file 3 Table S1'))
    df=pd.DataFrame(rows); assert len(df)==28
    q=pd.read_excel(src/'12915_2025_2219_MOESM2_ESM.xlsx',sheet_name='Fig. S3',header=None)
    control=q.iloc[1:4,1].astype(float).mean(); qp=[]
    for category,col in [('high_ORF',9),('low_ORF',10),('accessibility',11),('high_UTR',12)]:
        values=q.iloc[1:4,col].astype(float).values
        qp.append(dict(id='TC006679_'+category,expression_rep1=values[0],expression_rep2=values[1],
                       expression_rep3=values[2],control_mean=control,knockdown_pct=100*(1-values.mean()/control),
                       n_biological_replicates=3,readout='RT-qPCR 3 days after injection; five larvae per replicate; Additional file 2 Fig. S3'))
    return df, pd.DataFrame(qp)
