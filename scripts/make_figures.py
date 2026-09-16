"""Generate all main-manuscript figures from the released data and predictions."""
from collections import Counter
import argparse, itertools, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix,roc_auc_score
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import StrMethodFormatter
import matplotlib.patheffects as path_effects
from common import ROOT, RESULTS, CONFIG
from figstyle import plt,NAVY,BLUE,TAN,SKY,PALE,BROWN,PALETTE,DIVERGE,SEQUENTIAL,panels,save

def contrast_errorbars(ax):
    """Black error bars with an opaque white halo stay visible inside every bar."""
    from matplotlib.container import ErrorbarContainer
    for container in ax.containers:
        if isinstance(container,ErrorbarContainer):
            _,caps,segments=container.lines
            for artist in list(caps)+list(segments):
                artist.set_color('#101820');artist.set_zorder(5)
                artist.set_linewidth(1.15)
                if hasattr(artist,'set_markeredgecolor'):
                    artist.set_markeredgecolor('#101820');artist.set_markeredgewidth(1.15)
                artist.set_path_effects([path_effects.Stroke(linewidth=3.2,foreground='white'),path_effects.Normal()])

F=ROOT/'figures'; F.mkdir(exist_ok=True) 
LABELS={k:v['label'] for k,v in CONFIG['models'].items()}


def dataset():
    b=pd.read_csv(RESULTS/'source_gene_split.csv');raw=pd.read_csv(ROOT/'data/raw/ibeetle/manifest.csv')
    topics=(ROOT/'data/raw/ibeetle/topics.txt').read_text().splitlines();ph=raw[topics].eq(1).any(axis=1);le=raw.dali11.gt(20)
    fig,ax=plt.subplots(2,3,figsize=(7.2,4.5));panels(ax)
    ax[0,0].hist(b.seq.str.len(),bins=40,color=NAVY);ax[0,0].set(xlabel='Sequence length (bases)',ylabel='Sequences')
    ax[0,1].hist(raw.dali11.dropna(),bins=np.arange(0,105,5),color=NAVY);ax[0,1].axvline(20,color=BLUE,ls='--');ax[0,1].set(xlabel='dali11 mortality (%)',ylabel='Observed reagents')
    thresholds=np.arange(0,101,5)
    ax[0,2].plot(thresholds,[100*(raw.dali11.gt(t)|ph).mean() for t in thresholds],color=NAVY,label='Mortality or phenotype')
    ax[0,2].plot(thresholds,[100*raw.dali11.gt(t).mean() for t in thresholds],color=BLUE,label='Mortality only')
    ax[0,2].set(xlabel='Mortality threshold (%)',ylabel='Selected records (%)',ylim=(0,103));ax[0,2].legend(loc='center right',bbox_to_anchor=(1,.72),fontsize=6.5,frameon=True,facecolor='white',edgecolor='#D5DEE8',framealpha=1)
    counts=raw.tc.value_counts();ax[1,0].hist(counts,bins=np.arange(.5,counts.max()+1.5),color=NAVY);ax[1,0].set(xlabel='Reagents per target gene',ylabel='Genes',yscale='log')
    ax[1,1].bar(['Mortality\nonly','Phenotype\nonly','Both'],[(le&~ph).sum(),(~le&ph).sum(),(le&ph).sum()],color=PALETTE[:3]);ax[1,1].set(ylabel='Positive reagents');ax[1,1].tick_params(axis='x',labelsize=7)
    ax[1,2].bar(['Positive','Background'],[int(b.label.sum()),int(b.label.eq(0).sum())],color=PALETTE[:2]);ax[1,2].set(ylabel='Sequences')
    for a in [ax[0,0],ax[0,1],ax[1,1],ax[1,2]]:a.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    save(fig,F/'fig3_dataset')
    (RESULTS/'dataset_figure_metadata.json').write_text(json.dumps(dict(total=len(b),raw_screen_records=len(raw),observed_dali11=int(raw.dali11.notna().sum()),
        mortality_only=int((le&~ph).sum()),phenotype_only=int((~le&ph).sum()),both=int((le&ph).sum())),indent=2))


def separability():
    b=pd.read_csv(RESULTS/'source_gene_split.csv');fig,ax=plt.subplots(2,3,figsize=(7.2,4.55));panels(ax)
    km={};di={}
    for lab,name,color,offset in [(1,'Positive',NAVY,-.18),(0,'Background',BLUE,.18)]:
        seq=b.loc[b.label.eq(lab),'seq'];length=seq.str.len();gc=seq.str.count('[GC]')/length
        ax[0,0].hist(length,bins=np.arange(125,626,20),histtype='step',density=True,color=color,label=name)
        ax[0,1].hist(gc,bins=np.linspace(0,1,36),histtype='step',density=True,color=color,label=name)
        ax[0,2].bar(np.arange(4)+offset,[np.mean(seq.str.count(c)/length) for c in 'ACGT'],width=.36,color=color,label=name)
        for k,store in [(2,di),(3,km)]:
            keys=[''.join(v) for v in itertools.product('ACGT',repeat=k)];values=dict.fromkeys(keys,0.)
            for s in seq:
                c=Counter(s[i:i+k] for i in range(len(s)-k+1))
                for key,count in c.items():values[key]+=count/(len(s)-k+1)
            store[lab]={key:v/len(seq) for key,v in values.items()}
    ax[0,0].set(xlabel='Length (bases)',ylabel='Density');ax[0,0].legend(fontsize=7)
    ax[0,1].set(xlabel='GC fraction',ylabel='Density');ax[0,1].legend(loc='upper right',fontsize=7); ax[0,2].set_xticks(range(4),list('ACGT'));ax[0,2].set(ylabel='Mean base fraction',ylim=(0,.42));ax[0,2].legend(loc='upper right',fontsize=7,frameon=True,facecolor='white',edgecolor='#D5DEE8',framealpha=1)
    keys=list(di[1]);values=[100*(di[1][k]-di[0][k]) for k in keys]
    ax[1,0].bar(keys,values,color=NAVY);ax[1,0].tick_params(axis='x',rotation=90,labelsize=6.5);ax[1,0].set(ylabel='Dinucleotide difference (pp)');ax[1,0].axhline(0,lw=.6,color='gray')
    enrich=sorted([(k,np.log2((km[1][k]+1e-8)/(km[0][k]+1e-8))) for k in km[1]],key=lambda x:x[1]);selected=enrich[:6]+enrich[-6:]
    ax[1,1].bar([x[0] for x in selected],[x[1] for x in selected],color=NAVY);ax[1,1].tick_params(axis='x',rotation=90,labelsize=6.5);ax[1,1].set(ylabel=r'3-mer log$_2$ frequency ratio');ax[1,1].axhline(0,lw=.6,color='gray')
    ax[1,2].bar(['Positive','Background'],[int(b.label.sum()),int(b.label.eq(0).sum())],color=PALETTE[:2]);ax[1,2].set(ylabel='Sequences');ax[1,2].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    pd.DataFrame(dict(dinucleotide=keys,difference_percentage_points=values)).to_csv(RESULTS/'dinucleotide_differences.csv',index=False)
    pd.DataFrame(enrich,columns=['kmer','log2_positive_background_ratio']).to_csv(RESULTS/'trinucleotide_ratios.csv',index=False)
    save(fig,F/'fig4_separability')


def roc():
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.1));panels(ax)
    for name,color in zip(['beetleformer','cnn_bilstm','single_scale_transformer'],PALETTE):
        d=pd.read_csv(RESULTS/(name+'_test.csv'));f,t,_=roc_curve(d.label,d.score);p,r,_=precision_recall_curve(d.label,d.score)
        ax[0].plot(f,t,color=color,label=f'{LABELS[name]} ({roc_auc_score(d.label,d.score):.3f})')
        ax[1].plot(r,p,color=color,label=LABELS[name])
    ax[0].plot([0,1],[0,1],color='gray',ls='--',lw=.7);ax[0].set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(0,1),ylim=(0,1.02));ax[0].legend(loc='lower right',fontsize=6.8)
    ax[1].set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02));ax[1].legend(loc='lower left',fontsize=7)
    save(fig,F/'fig5_roc_pr')


def robustness():
    stats=json.loads((RESULTS/'manuscript_statistics.json').read_text());fig,ax=plt.subplots(1,2,figsize=(7.2,3.5));panels(ax)
    names=['beetleformer','without_strand_symmetry','without_convolutions','without_descriptors','without_attention_pool']
    labels=['BeetleFormer','No strand symmetry','No local convolutions','No descriptors','No attention pooling']
    for i,(name,color) in enumerate(zip(names,[NAVY,BLUE,TAN,SKY,BROWN])):
        v=stats[name]['AUC'];lo,hi=stats[name]['AUC_gene_bootstrap_95CI']
        ax[0].barh(i,v,height=.6,color=color,xerr=[[v-lo],[hi-v]],error_kw=dict(ecolor=NAVY,capsize=3,lw=.9))
        ax[0].text(.03,i,f'{v:.3f}',va='center',color='white' if color in [NAVY,BLUE,BROWN] else NAVY,fontsize=7.5)
    ax[0].set_yticks(range(len(names)),labels,fontsize=7);ax[0].invert_yaxis();ax[0].set(xlabel='Test ROC AUC (95% interval)',xlim=(0,1.04))
    names=['beetleformer','cnn_bilstm','features_only','composition_length','length_only'];x=np.arange(5)
    ax[1].bar(x-.18,[stats[n]['AUC'] for n in names],width=.36,color=NAVY,label='Full test set')
    ax[1].bar(x+.18,[stats[n]['exact_length_matched']['AUC'] for n in names],width=.36,color=BLUE,label='Exact-length matched')
    ax[1].set_xticks(x,['BeetleFormer','CNN–BiLSTM','Descriptors','Composition','Length'],rotation=35,ha='right',fontsize=7)
    ax[1].set(ylabel='ROC AUC',ylim=(0,1.23),yticks=np.arange(0,1.01,.2));ax[1].legend(loc='upper right',fontsize=7,frameon=True,facecolor='white',edgecolor='#D5DEE8',framealpha=1);contrast_errorbars(ax[0]);save(fig,F/'fig6_ablation_robustness')


def external():
    stats=json.loads((RESULTS/'external_statistics.json').read_text())
    df=pd.read_csv(RESULTS/'external_classification.csv')
    fig,ax=plt.subplots(1,2,figsize=(7.2,3.2));panels(ax)
    names=['beetleformer','cnn_bilstm','single_scale_transformer']
    for name,color in zip(names,[NAVY,BLUE,TAN]):
        fpr,tpr,_=roc_curve(df.label,df[name+'_logit'])
        ax[0].plot(fpr,tpr,color=color,label=CONFIG['models'][name]['label']+' ('+f"{stats[name]['AUC']:.3f}"+')')
    ax[0].plot([0,1],[0,1],color='gray',ls=':',lw=.8)
    ax[0].set(xlabel='False positive rate',ylabel='True positive rate',xlim=(-.02,1.02),ylim=(-.02,1.04))
    ax[0].legend(loc='lower right',fontsize=6.8,frameon=True,facecolor='white',edgecolor='#D5DEE8',framealpha=1)
    for i,(name,color) in enumerate(zip(names,[NAVY,BLUE,TAN])):
        value=stats[name]['Acc']
        ax[1].bar(i,value,width=.62,color=color)
        ax[1].text(i,value+.022,f'{value:.3f}',ha='center',fontsize=8)
    ax[1].set_xticks([0,1,2],['BeetleFormer','Adapted\nCNN–BiLSTM','Single-scale\nTransformer'],fontsize=7)
    ax[1].set(ylabel='External accuracy',ylim=(0,1.04),yticks=np.arange(0,1.01,.2),xlim=(-.6,2.6))
    save(fig,F/'fig7_external')


def interpretation():
    fig,ax=plt.subplots(2,2,figsize=(7.2,4.65));panels(ax)
    d=pd.read_csv(RESULTS/'interpretation_base_preference.csv');v=d[list('ACGT')].to_numpy().T;bound=np.abs(v).max()
    im=ax[0,0].imshow(v,aspect='auto',cmap=DIVERGE,vmin=-bound,vmax=bound,extent=(.5,60.5,3.5,-.5));ax[0,0].set_yticks(range(4),list('ACGT'));ax[0,0].set(xlabel='Position from supplied 5′ end');fig.colorbar(im,ax=ax[0,0],label='Positive − background',fraction=.04)
    d=pd.read_csv(RESULTS/'interpretation_saliency.csv');ax[0,1].plot(d.relative_position,d['mean'],color=NAVY);ax[0,1].set(xlabel='Relative sequence position',ylabel='Mean normalized saliency')
    d=pd.read_csv(RESULTS/'interpretation_pca.csv')
    for lab,color in [(1,NAVY),(0,BLUE)]:
        g=d[d.label.eq(lab)];ax[1,0].scatter(g.PC1,g.PC2,s=3,alpha=.45,color=color,label='Positive' if lab else 'Background',rasterized=True)
    ax[1,0].set(xlabel='Principal component 1',ylabel='Principal component 2')
    legend=ax[1,0].legend(loc='lower left',fontsize=7,markerscale=2,frameon=True,facecolor='white',edgecolor='#D5DEE8',framealpha=1)
    for handle in legend.legend_handles:handle.set_alpha(1)
    d=pd.read_csv(RESULTS/'beetleformer_test.csv');c=confusion_matrix(d.label,d.score>=.5);ax[1,1].imshow(c,cmap=SEQUENTIAL,vmin=0,vmax=c.max())
    for i,j in itertools.product(range(2),repeat=2):ax[1,1].text(j,i,f'{c[i,j]:,}',ha='center',va='center',color='white' if c[i,j]>.5*c.max() else 'black',fontsize=10)
    ax[1,1].set_xticks([0,1],['Background','Positive']);ax[1,1].set_yticks([0,1],['Background','Positive']);ax[1,1].set(xlabel='Predicted class',ylabel='Reference class');save(fig,F/'fig8_interpretability')


def length():
    split=pd.read_csv(RESULTS/'source_gene_split.csv');d=pd.read_csv(RESULTS/'beetleformer_test.csv').merge(split[['id','seq']],on='id');d['length']=d.seq.str.len()
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.95));panels(ax)
    values=[roc_auc_score(g.label,g.score) for lo,hi in [(130,399),(400,615)] for g in [d[d.length.between(lo,hi)]]]
    ax[0].bar(['130–399','400–615'],values,color=PALETTE[:2],width=.55);ax[0].set(xlabel='Sequence length (bases)',ylabel='ROC AUC',ylim=(0,1.06))
    for i,v in enumerate(values):ax[0].text(i,v+.015,f'{v:.3f}',ha='center',fontsize=8)
    for lab,color in [(1,NAVY),(0,BLUE)]:
        g=d[d.label.eq(lab)];ax[1].scatter(g.length,g.score,s=2,alpha=.2,color=color,label='Positive' if lab else 'Background',rasterized=True)
    ax[1].set(xlabel='Sequence length (bases)',ylabel='Classification score')
    legend=ax[1].legend(loc='center left',fontsize=7,markerscale=3,frameon=True,facecolor='white',edgecolor='#D5DEE8',framealpha=1)
    for handle in legend.legend_handles:handle.set_alpha(1)
    save(fig,F/'fig9_length')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--figures',nargs='+',default=['dataset','separability','roc','robustness','external','interpretation','length']);args=ap.parse_args()
    for name in args.figures:globals()[name]();print('Generated',name,flush=True)
