import copy,json,unittest,xml.etree.ElementTree as ET
from fractions import Fraction as F
from pathlib import Path
from banditlab.demo import demo
from examples.render_audit import render
NS={'s':'http://www.w3.org/2000/svg'}

class FigureTests(unittest.TestCase):
    def test_all28_vertices_and_shared_scale(self):
        d=demo();root=ET.fromstring(render(d));curves={n.attrib['data-curve']:n for n in root.findall('s:polyline',NS)};self.assertEqual(set(curves),{'value','greedy_value','best_fixed_arm_value','clairvoyant'})
        for key,node in curves.items():
            coords=[tuple(map(float,p.split(','))) for p in node.attrib['points'].split()];expected=[(0,F(0))]+[(r['horizon'],F(*(r[key] if key!='clairvoyant' else r['clairvoyant']['total_expected_reward_upper_bound']))) for r in d['runs']];self.assertEqual(len(coords),7)
            for (x,y),(h,v) in zip(coords,expected):self.assertAlmostEqual(x,90+70*h,places=8);self.assertAlmostEqual(y,float(520-60*v),places=8)
        self.assertIn('stroke-dasharray',curves['clairvoyant'].attrib)
    def test_full_metadata_and_tie_scope(self):
        root=ET.fromstring(render(demo()));self.assertEqual(json.loads(root.find('s:metadata',NS).text),demo());text=' '.join(root.itertext());self.assertIn('BOTH first arms tie',text);self.assertIn('not future reward draws',text)
    def test_regeneration(self):self.assertEqual((Path(__file__).resolve().parents[1]/'assets'/'bandit-audit.svg').read_text(),render(demo()))
    def test_reject_undisplayed_state_or_cdf_change(self):
        for location in ['state','cdf']:
            d=copy.deepcopy(demo())
            if location=='state':d['runs'][-1]['states'][0]['posterior'][0][0]+=1
            else:d['runs'][-1]['clairvoyant']['max_parameter_cdf_polynomial'][0][0]+=1
            with self.assertRaises(ValueError):render(d)
