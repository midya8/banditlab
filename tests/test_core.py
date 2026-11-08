import copy,itertools,json,random,subprocess,sys,tempfile,unittest
from fractions import Fraction as F
from math import comb,factorial
from pathlib import Path
from unittest.mock import patch
from banditlab import solve
from banditlab.core import _cdf,_clairvoyant_mean


def f(x):return F(*x)
def pack(x):return [x.numerator,x.denominator]
def trees(k,h):
    if h==0:return [None]
    smaller=trees(k,h-1)
    return [(a,zero,one) for a in range(k) for zero in smaller for one in smaller]
def integrated_return(priors,tree,h):
    # Enumerate complete binary outcome strings and integrate latent theta directly.
    # Beta moments replace production posterior-predictive transitions/DP.
    value=F(0)
    for outcomes in itertools.product([0,1],repeat=h):
        counts=[[0,0] for _ in priors];node=tree
        for r in outcomes:
            a=node[0];counts[a][0 if r else 1]+=1;node=node[2 if r else 1]
        probability=F(1)
        for (a,b),(s,z) in zip(priors,counts):
            probability*=F(factorial(a+s-1)*factorial(b+z-1)*factorial(a+b-1),factorial(a+b+s+z-1)*factorial(a-1)*factorial(b-1))
        value+=sum(outcomes)*probability
    return value

def greedy_tree(priors,h):
    if not h:return None
    means=[F(a,a+b) for a,b in priors];a=max(range(len(priors)),key=lambda i:means[i]);children=[]
    for r in [0,1]:
        nxt=copy.deepcopy(priors);nxt[a][0 if r else 1]+=1;children.append(greedy_tree(nxt,h-1))
    return (a,*children)

class CoreTests(unittest.TestCase):
    def test_all2048_complete_depth3_policies_16_prior_pairs(self):
        count=0
        for priors in itertools.product([[1,1],[1,2],[2,1],[2,2]],repeat=2):
            priors=[p[:] for p in priors];refs=[(t,integrated_return(priors,t,3)) for t in trees(2,3)];r=solve(priors,horizon=3)
            best=max(v for _,v in refs);q=[max(v for t,v in refs if t[0]==a) for a in range(2)]
            self.assertEqual(r['value'],pack(best));self.assertEqual(r['action_values'],list(map(pack,q)));self.assertEqual(r['optimal_actions'],[a for a,v in enumerate(q) if v==best]);self.assertEqual(r['greedy_value'],pack(integrated_return(priors,greedy_tree(priors,3),3)));count+=len(refs)
        self.assertEqual(count,2048)

    def test_1728_three_arm_depth2_policy_references(self):
        count=0
        for priors in itertools.product([[1,1],[1,2],[2,1],[2,2]],repeat=3):
            priors=[p[:] for p in priors];refs=[(t,integrated_return(priors,t,2)) for t in trees(3,2)];r=solve(priors,horizon=2);best=max(v for _,v in refs);q=[max(v for t,v in refs if t[0]==a) for a in range(3)]
            self.assertEqual(r['value'],pack(best));self.assertEqual(r['action_values'],list(map(pack,q)));self.assertEqual(r['optimal_actions'],[a for a,v in enumerate(q) if v==best]);count+=len(refs)
        self.assertEqual(count,1728)

    def test_every_state_edge_q_tie_and_greedy_value_reconstructed(self):
        r=solve([[2,3],[4,2],[1,1]],horizon=5);states=r['states'];self.assertEqual(len(states),comb(11,6));seen=set()
        for n in states:
            self.assertEqual(n['id'],len(seen));seen.add(tuple(n['increments']));self.assertEqual(sum(n['increments']),n['elapsed']);self.assertEqual(n['remaining']+n['elapsed'],5)
            means=[F(a,a+b) for a,b in n['posterior']];self.assertEqual(n['posterior_means'],list(map(pack,means)));self.assertEqual(n['best_fixed_arm_value'],pack(n['remaining']*max(means)))
            if not n['remaining']:
                self.assertEqual(n['value'],[0,1]);self.assertEqual(n['greedy_value'],[0,1]);self.assertIsNone(n['selected_action']);self.assertEqual(n['branches'],[]);continue
            qs=[];gqs=[]
            for branch in n['branches']:
                a=branch['action'];s=states[branch['success']['state_id']];z=states[branch['failure']['state_id']]
                for reward,ch in [(1,s),(0,z)]:
                    self.assertLess(ch['id'],n['id']);self.assertEqual(ch['remaining'],n['remaining']-1);expected=copy.deepcopy(n['posterior']);expected[a][0 if reward else 1]+=1;self.assertEqual(ch['posterior'],expected)
                self.assertEqual(branch['success']['probability'],pack(means[a]));self.assertEqual(branch['failure']['probability'],pack(1-means[a]));qs.append(means[a]*(1+f(s['value']))+(1-means[a])*f(z['value']));gqs.append(means[a]*(1+f(s['greedy_value']))+(1-means[a])*f(z['greedy_value']))
            self.assertEqual(n['action_values'],list(map(pack,qs)));self.assertEqual(n['greedy_continuation_action_values'],list(map(pack,gqs)));self.assertEqual(n['value'],pack(max(qs)));self.assertEqual(n['optimal_actions'],[a for a,v in enumerate(qs) if v==max(qs)]);self.assertEqual(n['selected_action'],n['optimal_actions'][0]);self.assertEqual(n['greedy_action'],means.index(max(means)));self.assertEqual(n['greedy_value'],pack(gqs[n['greedy_action']]))
        self.assertEqual(len(seen),len(states));self.assertEqual(r['root_state_id'],len(states)-1)

    def test_exact_state_and_edge_counts_at_caps(self):
        r=solve([[1,1],[1,1]],horizon=12,state_limit=1820);self.assertEqual(r['counts'],{'preflight_states':1820,'returned_states':1820,'terminal_states':455,'decision_states':1365,'action_outcome_edges':5460,'state_limit':1820})
        r=solve([[1,1]]*3,horizon=8,state_limit=3003);self.assertEqual(r['counts']['returned_states'],3003);self.assertEqual(r['counts']['action_outcome_edges'],6*comb(13,6))

    def test_preflight_before_state_generation_or_polynomial(self):
        with patch('banditlab.core._compositions',side_effect=AssertionError('allocated')),patch('banditlab.core._clairvoyant_mean',side_effect=AssertionError('integrated')):
            for priors,h,cap in [([[1,1],[1,1]],12,1819),([[1,1]]*3,12,4096)]:
                with self.assertRaisesRegex(ValueError,'preflight refused'):solve(priors,horizon=h,state_limit=cap)

    def test_horizon_zero_and_one_full_ties(self):
        r=solve([[1,1],[2,2],[3,3]],horizon=0,state_limit=1);self.assertEqual(r['value'],[0,1]);self.assertEqual(r['optimal_actions'],[]);self.assertIsNone(r['selected_action']);self.assertEqual(r['best_fixed_arms'],[]);self.assertEqual(r['clairvoyant']['optimal_bayes_regret'],[0,1])
        r=solve([[1,1],[2,2],[3,3]],horizon=1);self.assertEqual(r['optimal_actions'],[0,1,2]);self.assertEqual(r['selected_action'],0);self.assertEqual(r['greedy_first_action'],0);self.assertEqual(r['value'],[1,2])

    def test_single_arm_exact_no_learning_gain_or_bayes_regret(self):
        for priors,h in [([[1,1]],12),([[16,1]],7),([[1,16]],4),([[16,16]],2)]:
            r=solve(priors,horizon=h);expected=F(h*priors[0][0],sum(priors[0]));self.assertEqual(r['value'],pack(expected));self.assertEqual(r['greedy_value'],pack(expected));self.assertEqual(r['best_fixed_arm_value'],pack(expected));self.assertEqual(r['clairvoyant']['optimal_bayes_regret'],[0,1])

    def test_cdf_binomial_reference_normalization_and_moments(self):
        for a,b in itertools.product(range(1,17),repeat=2):
            c=_cdf(a,b);self.assertEqual(sum(c),1);self.assertEqual(c[0],0);mean=1-sum(x/F(i+1) for i,x in enumerate(c));self.assertEqual(mean,F(a,a+b))
            # Bernstein-binomial CDF reference at rational points, independently from monomial density expansion.
            for x in [F(0),F(1,7),F(1,2),F(1)]:
                reference=sum(F(comb(a+b-1,j))*x**j*(1-x)**(a+b-1-j) for j in range(a,a+b));self.assertEqual(sum(v*x**i for i,v in enumerate(c)),reference)

    def test_expected_max_integral_alternate_bernstein_reference(self):
        # Product CDF integration in Bernstein basis, not production monomial multiplication.
        from math import prod
        for priors in [[[1,1]]*k for k in [1,2,3]]+[[[2,3],[4,2]],[[16,16],[16,16],[16,16]],[[1,1],[5,4]]]:
            terms=[[(j, a+b-1-j,comb(a+b-1,j)) for j in range(a,a+b)] for a,b in priors];cdf_integral=F(0)
            for choices in itertools.product(*terms):
                x=sum(t[0] for t in choices);y=sum(t[1] for t in choices);cdf_integral+=F(prod(t[2] for t in choices)*factorial(x)*factorial(y),factorial(x+y+1))
            mean,polynomial=_clairvoyant_mean(priors);self.assertEqual(mean,1-cdf_integral);self.assertEqual(sum(polynomial),1)
            if priors==[[1,1]]*len(priors):self.assertEqual(mean,F(len(priors),len(priors)+1))

    def test_exploration_fixture_and_horizon_two_tie_not_strict_gain(self):
        priors=[[1,1],[5,4]];r=solve(priors,horizon=3);self.assertEqual(r['optimal_actions'],[0]);self.assertEqual(r['greedy_first_action'],1);self.assertEqual(r['value'],[187,108]);self.assertEqual(r['greedy_value'],[46,27]);self.assertEqual(r['adaptive_gain_over_greedy'],[1,36]);self.assertEqual(r['best_fixed_arm_value'],[5,3]);self.assertEqual(r['clairvoyant']['mean_best_parameter'],[2,3]);self.assertEqual(r['clairvoyant']['optimal_bayes_regret'],[29,108])
        r=solve(priors,horizon=2);self.assertEqual(r['optimal_actions'],[0,1]);self.assertEqual(r['adaptive_gain_over_greedy'],[0,1])

    def test_priors_are_not_resampled_on_each_pull(self):
        r=solve([[1,1],[5,4]],horizon=3);root=r['states'][-1];branch=root['branches'][0];success=r['states'][branch['success']['state_id']];failure=r['states'][branch['failure']['state_id']]
        self.assertEqual(success['posterior'][0],[2,1]);self.assertEqual(failure['posterior'][0],[1,2]);self.assertEqual(success['selected_action'],0);self.assertEqual(failure['selected_action'],1)

    def test_validation_strict_types_shapes_and_caps(self):
        for priors in [None,(),[],[[1,1]]*4,[[1]],[[0,1]],[[1,17]],[[True,1]],[[1.0,1]],[[1,'1']],[[1,1],None]]:
            with self.assertRaises(ValueError):solve(priors)
        for h in [-1,13,True,1.0,None]:
            with self.assertRaises(ValueError):solve([[1,1]],horizon=h)
        for c in [0,20001,True,1.5,None]:
            with self.assertRaises(ValueError):solve([[1,1]],state_limit=c)

    def test_input_output_rng_and_repeatability(self):
        priors=[[1,1],[5,4]];before=copy.deepcopy(priors);state=random.getstate();r=solve(priors);expected=copy.deepcopy(r);r['priors'][0][0]=999;r['states'][-1]['posterior'][0][0]=999
        self.assertEqual(priors,before);self.assertEqual(random.getstate(),state);self.assertEqual(solve(priors),expected)

    def test_cli_matches_api(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.json';path.write_text(json.dumps({'version':1,'priors':[[1,1],[5,4]],'horizon':3}));r=subprocess.run([sys.executable,'-m','banditlab','solve',str(path)],capture_output=True,text=True,check=True);self.assertEqual(json.loads(r.stdout),solve([[1,1],[5,4]],horizon=3))

    def test_cli_errors_duplicate_extra_oversize_bad_missing(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.json'
            for raw in ['{','{"version":1,"version":1}',' '*65537,json.dumps({'version':1,'priors':[[1,1]],'horizon':3,'extra':1}),json.dumps({'version':True,'priors':[[1,1]],'horizon':3})]:
                path.write_text(raw);r=subprocess.run([sys.executable,'-m','banditlab','solve',str(path)],capture_output=True,text=True);self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'');self.assertIn('error:',r.stderr)
            path.unlink();r=subprocess.run([sys.executable,'-m','banditlab','solve',str(path)],capture_output=True,text=True);self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'')

if __name__=='__main__':unittest.main()
