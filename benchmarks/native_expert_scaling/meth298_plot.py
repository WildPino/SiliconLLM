#!/usr/bin/env python3
"""Plot the complete fixed-rank actual-input screen without changing gates."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
path=DOC/'meth298_gigachat_full_covariance_result.json'
assert hashlib.sha256(path.read_bytes()).hexdigest()=='b0b9a4773b02dc6c309504f2eddc9a8dcc623d115d70c751df9d392219d00223'
r=json.loads(path.read_text(encoding='utf-8'))
fig,axes=plt.subplots(1,3,figsize=(13,4.4),sharey=True)
curves=[('test_ordinary_energy','Ordinary fit','#999999','--'),
        ('test_diagonal_energy','Diagonal fit','#527ca8','-.'),
        ('test_full_covariance_energy','Full covariance fit','#08786e','-'),
        ('test_oracle_energy','Test-domain oracle','#c17b22',':')]
for ax,l in zip(axes,(1,13,25)):
    rows=[x for x in r['rows'] if x['layer']==l];assert len(rows)==9
    labels=[str(x['expert'])+'/'+{'gate_proj':'G','up_proj':'U','down_proj':'D'}[x['projection']] for x in rows]
    for key,label,color,style in curves:
        ax.plot(range(9),[100*x[key] for x in rows],label=label,color=color,linestyle=style,marker='o',markersize=3,linewidth=1.5)
    ax.axhline(90,color='#a74242',linewidth=1,alpha=.7)
    ax.set_title(f'Base layer {l}');ax.set_xticks(range(9),labels,rotation=45)
    ax.set_ylim(30,100);ax.set_xlabel('Expert / projection');ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Actual test output energy retained (%)')
fig.suptitle('GigaChat rank192: full covariance median66.43%; all four fixed gates fail',fontsize=12)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.035),frameon=False)
fig.text(.5,.008,'All27 projections; FP64 factors; source-BF16 routed inputs. Red line: every-row90% floor. Oracle is diagnostic only.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.13,1,.93))
out=DOC/'meth298_actual_input_comparison.png';assert not out.exists()
fig.savefig(out,dpi=160);plt.close(fig)
print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
