import base64,copy,json,unittest,zlib,xml.etree.ElementTree as ET
from fractions import Fraction as F
from pathlib import Path
from banditlab.audit_demo import demo
from examples.render_policy_audit import render
NS={'s':'http://www.w3.org/2000/svg'}

class PolicyFigureTests(unittest.TestCase):
    def test_all18_bars_exact_positions_and_two_metric_scales(self):
        d=demo();root=ET.fromstring(render(d));bars=[n for n in root.findall('s:rect',NS) if 'data-metric' in n.attrib];self.assertEqual(len(bars),18)
        for n in bars:
            name=n.attrib['data-policy'];i=int(n.attrib['data-pull'])-1;j=['optimal','greedy','fixed'].index(name);step=d['policies'][name]['per_pull'][i];arm=n.attrib['data-metric']=='arm0';v=F(*(step['arm_probability_mass'][0] if arm else step['weighted_optimal_bellman_gap']));scale=240 if arm else 7200
            self.assertAlmostEqual(float(n.attrib['height']),float(scale*v),places=8);self.assertAlmostEqual(float(n.attrib['y']),float(480-scale*v),places=8);self.assertEqual(int(n.attrib['x']),(110+140*i+28*j) if arm else (695+130*i+28*j))
    def test_metadata_complete_and_scope(self):
        root=ET.fromstring(render(demo()));self.assertEqual(json.loads(zlib.decompress(base64.b64decode(root.find('s:metadata',NS).text))),demo());text=' '.join(root.itertext());self.assertIn('not history counts',text);self.assertIn('Summed gaps equal policy loss',text)
    def test_regeneration(self):self.assertEqual((Path(__file__).resolve().parents[1]/'assets'/'policy-audit.svg').read_text(),render(demo()))
    def test_undisplayed_zero_mass_conditional_value_tamper_rejected(self):
        d=copy.deepcopy(demo());r=d['policies']['fixed'];n=next(n for n in r['states'] if n['remaining'] and n['occupancy']==[0,1]);n['conditional_policy_value'][0]+=1
        with self.assertRaises(ValueError):render(d)
