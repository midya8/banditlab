"""Reproducible policy occupancy and performance-difference panels."""
import argparse,base64,html,json,zlib
from fractions import Fraction as F
from pathlib import Path
from banditlab.audit_demo import demo


def render(d):
    if d!=demo():raise ValueError('requires complete unchanged policy audit demo')
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="820" viewBox="0 0 1120 820">','<rect width="1120" height="820" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>','<metadata data-encoding="zlib+base64">'+base64.b64encode(zlib.compress(json.dumps(d,sort_keys=True,separators=(',',':')).encode())).decode()+'</metadata>']
    def text(x,y,s,size=16,color='#e7edf8'):
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}">'+html.escape(s)+'</text>')
    def coord(x):return format(float(x),'.9f').rstrip('0').rstrip('.')
    text(30,44,'BanditLab | where a policy goes, and where its value is lost',27)
    text(30,80,'Three pulls. Fixed latent Beta(1,1) / Beta(5,4) arm rates. Exact supplied-prior flows.',17)
    text(70,130,'Probability of choosing exploratory arm 0',20)
    text(620,130,'Occupancy-weighted optimal Bellman gap',20)
    colors=['#63d8bc','#e9bb79','#98a8c0'];names=['optimal','greedy','fixed']
    for x,label,color in zip([70,310,550],['Optimal adaptive','Updating greedy','Fixed arm 1'],colors):
        out.append(f'<rect x="{x}" y="162" width="18" height="12" fill="{color}"/>');text(x+27,175,label,17)
    for v in [F(0),F(1,2),F(1)]:
        y=480-240*v;out.append(f'<path d="M 75 {coord(y)} H 500" stroke="#33445e"/>');text(36,y+5,str(v),15,'#b7c4d9')
    for v in [F(0),F(1,60),F(1,30)]:
        y=480-7200*v;out.append(f'<path d="M 670 {coord(y)} H 1090" stroke="#33445e"/>');text(610,y+5,str(v),15,'#b7c4d9')
    for i in range(3):
        for j,name in enumerate(names):
            step=d['policies'][name]['per_pull'][i]
            for metric,key,x,scale in [('arm0','arm_probability_mass',110+140*i+28*j,240),('gap','weighted_optimal_bellman_gap',695+130*i+28*j,7200)]:
                value=F(*(step[key][0] if metric=='arm0' else step[key]));height=scale*value
                out.append(f'<rect data-metric="{metric}" data-policy="{name}" data-pull="{i+1}" x="{x}" y="{coord(480-height)}" width="23" height="{coord(height)}" fill="{colors[j]}"/>')
        text(123+140*i,512,'Pull '+str(i+1),17);text(708+130*i,512,'Pull '+str(i+1),17)
    text(70,553,'Arm 0 mass by pull: optimal 1, 1/2, 1/3;',17,'#63d8bc')
    text(70,580,'greedy 0, 4/9, 2/9; fixed 0, 0, 0.',17)
    text(620,553,'Greedy loss: 1/36 + 0 + 0 = 1/36.',17,'#e9bb79')
    text(620,580,'Fixed loss: 1/36 + 8/297 + 1/99 = 7/108.',17)
    text(30,620,'Policy',17);text(295,620,'Expected total reward',17);text(585,620,'Expected arm 0 pulls',17);text(850,620,'Positive-mass states',17);text(850,643,'/ full graph states',15,'#b7c4d9')
    for y,name,label in [(667,'optimal','Optimal'),(700,'greedy','Greedy'),(733,'fixed','Fixed')]:
        r=d['policies'][name];text(30,y,label,18,colors[names.index(name)]);text(295,y,str(F(*r['expected_reward'])),18);text(585,y,str(F(*r['expected_pulls_by_arm'][0])),18);text(850,y,str(r['counts']['positive_mass_states'])+' / '+str(r['counts']['full_graph_states']),18)
    text(30,777,'Gap = optimal state value minus chosen-action optimal-continuation Q. Summed gaps equal policy loss.',16)
    text(30,805,'Occupancy sums history probabilities, not history counts. Not empirical trials or a causal / frequentist guarantee.',16,'#b7c4d9')
    return '\n'.join(out+['</svg>'])+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();s=render(demo());path=Path(__file__).resolve().parents[1]/'assets'/'policy-audit.svg'
    if a.check:
        if path.read_text()!=s:raise SystemExit('SVG differs')
    else:path.parent.mkdir(exist_ok=True);path.write_text(s)
