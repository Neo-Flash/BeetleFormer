"""Original manuscript palette; current user-specified category priority."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

NAVY='#0F2B5F';BLUE='#5991C7';TAN='#D5A370';SKY='#8EC1E7';PALE='#B9DBF1';BROWN='#7B4F37'
PALETTE=[NAVY,BLUE,TAN,SKY,PALE,BROWN]
DIVERGE=LinearSegmentedColormap.from_list('beetle_div',[BROWN,TAN,'white',SKY,NAVY])
SEQUENTIAL=LinearSegmentedColormap.from_list('beetle_seq',['white',PALE,SKY,BLUE,NAVY])
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':8.5,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
    'legend.fontsize':7.5,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.8,'lines.linewidth':1.4,'pdf.fonttype':42,'svg.fonttype':'none',
    'savefig.dpi':600,'figure.dpi':120,'figure.constrained_layout.use':True})


def panels(axes):
    import numpy as np
    for i,ax in enumerate(np.array(axes).ravel()):
        ax.text(-.14,1.035,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=11)


def save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(path.with_suffix('.'+ext),dpi=600,facecolor='white')
    plt.close(fig)
    with Image.open(path.with_suffix('.png')) as im:
        im.convert('RGB').save(path.with_suffix('.png'),dpi=(600,600))
