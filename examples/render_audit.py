"""Exact expected-reward curves and a horizon-three exploration witness."""
import argparse,html,json
from fractions import Fraction as F
from pathlib import Path
from banditlab.demo import demo


def render(d):
    if d!=demo():raise ValueError('requires complete unchanged bandit demo')
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="750" viewBox="0 0 1120 750">','<rect width="1120" height="750" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>','<metadata>'+html.escape(json.dumps(d,sort_keys=True))+'</metadata>']
    def text(x,y,s,size=16,color='#e7edf8'):
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}">'+html.escape(s)+'</text>')
    def coord(x):return format(float(x),'.9f').rstrip('0').rstrip('.')
    text(30,43,'BanditLab | learn before committing to the higher-mean arm',27)
    text(30,81,'Independent latent rates: arm 0 ~ Beta(1,1), arm 1 ~ Beta(5,4). Exact prior expectations.',17)
    text(30,126,'Expected total unit rewards over the SAME pull budget',19)
    for v in [0,2,4,6]:
        y=520-60*v;out.append(f'<path d="M 90 {y} H 650" stroke="#33445e"/>');text(60,y+5,str(v),16,'#b7c4d9')
    for h in [0,2,4,6,8]:text(84+70*h,548,str(h),16,'#b7c4d9')
    for key,label,color,dash in [('best_fixed_arm_value','Best fixed arm','#98a8c0',False),('greedy_value','Posterior-mean greedy','#e9bb79',False),('value','Optimal adaptive','#63d8bc',False),('clairvoyant','Clairvoyant upper bound','#c4a2ff',True)]:
        pts=[(F(0),F(0))]+[(F(r['horizon']),F(*(r[key] if key!='clairvoyant' else r['clairvoyant']['total_expected_reward_upper_bound']))) for r in d['runs']]
        coords=' '.join(coord(90+70*x)+','+coord(520-60*y) for x,y in pts)
        out.append(f'<polyline data-curve="{key}" points="{coords}" fill="none" stroke="{color}" stroke-width="3"'+(' stroke-dasharray="8 5"' if dash else '')+'/>')
    text(90,586,'Budget (pulls). Lines connect the six evaluated budgets, plus zero.',16)
    for y,label,color in [(160,'Green: optimal adaptive','#63d8bc'),(193,'Amber: posterior-mean greedy','#e9bb79'),(226,'Gray: best fixed-arm commitment','#98a8c0'),(259,'Dashed: knows latent rates','#c4a2ff')]:text(730,y,label,16,color)
    text(730,304,'Three pulls: exploration strictly wins',19)
    text(730,338,'Prior means: 1/2 < 5/9.',18)
    text(730,372,'Adaptive first arm: 0. Greedy: 1.',18)
    text(730,408,'Adaptive value: 187/108.',18,'#63d8bc')
    text(730,441,'Greedy value: 46/27. Gain: 1/36.',18,'#e9bb79')
    text(730,477,'Fixed value: 5/3. Upper bound: 2.',18)
    text(730,513,'After arm 0 succeeds: choose arm 0.',16)
    text(730,544,'After arm 0 fails: choose arm 1.',16)
    text(30,636,'At one pull, arm 1 is optimal. At two pulls, BOTH first arms tie and the adaptive-greedy gain is zero.',17)
    text(30,674,'Greedy updates its posterior after every reward. The upper bound knows rates, not future reward draws.',17)
    text(30,714,'Fixed independent Beta-Bernoulli model only. Not empirical trials, frequentist regret or a universal greedy ranking.',16,'#b7c4d9')
    return '\n'.join(out+['</svg>'])+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();s=render(demo());path=Path(__file__).resolve().parents[1]/'assets'/'bandit-audit.svg'
    if a.check:
        if path.read_text()!=s:raise SystemExit('SVG differs')
    else:path.parent.mkdir(exist_ok=True);path.write_text(s)
