import copy,itertools,json,random,subprocess,sys,tempfile,unittest
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from math import factorial
from pathlib import Path
from unittest.mock import patch
from banditlab import solve,audit_policy
from test_core import trees,integrated_return,pack


def moment(priors,inc):
    p=F(1)
    for i,(a,b) in enumerate(priors):
        s,z=inc[2*i:2*i+2]
        p*=F(factorial(a+s-1)*factorial(b+z-1)*factorial(a+b-1),factorial(a+b+s+z-1)*factorial(a-1)*factorial(b-1))
    return p

@lru_cache(None)
def independent_best(priors,h):
    refs=[(t,integrated_return(priors,t,h)) for t in trees(len(priors),h)]
    best=max(v for _,v in refs)
    return next(t[0] for t,v in refs if v==best)


def history_reference(priors,h,weights):
    # Enumerate joint action/reward prefixes. Integrate theta moments directly.
    # No production probability, graph edge or backward value recursion is used.
    k=len(priors);occupancy=defaultdict(F);arm=[[F(0)]*k for _ in range(h)];success=[[F(0)]*k for _ in range(h)]
    def walk(inc,t,coins):
        occupancy[tuple(inc)]+=coins*moment(priors,inc)
        if t==h:return
        for a,w in enumerate(weights(inc,t)):
            arm[t][a]+=coins*w*moment(priors,inc)
            for r in [0,1]:
                nxt=inc[:];nxt[2*a+(0 if r else 1)]+=1
                if r:success[t][a]+=coins*w*moment(priors,nxt)
                walk(nxt,t+1,coins*w)
    walk([0]*(2*k),0,F(1))
    return occupancy,arm,success


class AuditTests(unittest.TestCase):
    def test_144_named_policy_history_references(self):
        count=0
        for priors in itertools.product([[1,1],[1,2],[2,1],[2,2]],repeat=2):
            priors=[p[:] for p in priors]
            for h in [1,2,3]:
                initial=max(range(2),key=lambda a:F(priors[a][0],sum(priors[a])))
                for name in ['optimal','greedy','fixed']:
                    def weights(inc,t):
                        post=tuple(tuple(priors[a][j]+inc[2*a+j] for j in range(2)) for a in range(2))
                        action=independent_best(post,h-t) if name=='optimal' else max(range(2),key=lambda a:F(post[a][0],sum(post[a]))) if name=='greedy' else initial
                        return [F(int(a==action)) for a in range(2)]
                    occ,arm,succ=history_reference(priors,h,weights);audit=audit_policy(priors,horizon=h,policy=name);solution=solve(priors,horizon=h)
                    for n,rec in zip(solution['states'],audit['states']):self.assertEqual(rec['occupancy'],pack(occ[tuple(n['increments'])]))
                    self.assertEqual(audit['expected_reward'],pack(sum(map(sum,succ))));self.assertEqual(audit['expected_pulls_by_arm'],[pack(sum(row[a] for row in arm)) for a in range(2)])
                    for step,am,sm in zip(audit['per_pull'],arm,succ):self.assertEqual(step['arm_probability_mass'],list(map(pack,am)));self.assertEqual(step['success_mass_by_arm'],list(map(pack,sm)))
                    self.assertEqual(F(*audit['policy_loss']),F(*solution['value'])-sum(map(sum,succ)));count+=1
        self.assertEqual(count,144)

    def test_randomized_kernels_joint_histories_2_and_3_arms(self):
        for priors,h in [([[1,1],[5,4]],4),([[2,3],[1,2],[3,1]],3)]:
            s=solve(priors,horizon=h);k=len(priors)
            def weights(inc,t):
                raw=[1+(sum((i+1)*v for i,v in enumerate(inc))+a+t)%4 for a in range(k)];return [F(x,sum(raw)) for x in raw]
            kernel=[list(map(pack,weights(n['increments'],n['elapsed']))) if n['remaining'] else [] for n in s['states']];r=audit_policy(priors,horizon=h,policy=kernel);occ,arm,succ=history_reference(priors,h,weights)
            self.assertEqual(r['expected_reward'],pack(sum(map(sum,succ))))
            for n,a in zip(s['states'],r['states']):self.assertEqual(a['occupancy'],pack(occ[tuple(n['increments'])]))
            for step,am,sm in zip(r['per_pull'],arm,succ):self.assertEqual(step['arm_probability_mass'],list(map(pack,am)));self.assertEqual(step['success_mass_by_arm'],list(map(pack,sm)))

    def test_every_flow_conservation_conditional_value_and_gap(self):
        s=solve([[2,1],[1,3],[2,2]],horizon=4);kernel=[[[1,6],[2,6],[3,6]] if n['remaining'] else [] for n in s['states']];r=audit_policy(s['priors'],horizon=4,policy=kernel);incoming=defaultdict(F);incoming[r['root_state_id']]=F(1)
        for n,a in zip(s['states'],r['states']):
            m=F(*a['occupancy']);reward=F(0);pv=F(0);gap=F(*n['value'])
            for flow in a['outgoing']:
                w=F(*flow['probability_mass']);incoming[flow['child_state_id']]+=w;reward+=flow['reward']*w
            for arm,pa in enumerate(a['action_probabilities']):
                pa=F(*pa);gap-=pa*F(*n['action_values'][arm])
                for reward,key in [(0,'failure'),(1,'success')]:
                    b=n['branches'][arm][key];pv+=pa*F(*b['probability'])*(reward+F(*r['states'][b['state_id']]['conditional_policy_value']))
            self.assertEqual(a['conditional_policy_value'],pack(pv));self.assertEqual(a['optimal_bellman_gap'],pack(gap));self.assertEqual(a['weighted_optimal_bellman_gap'],pack(m*gap));self.assertEqual(sum(F(*x['probability_mass']) for x in a['outgoing']),m if n['remaining'] else 0)
        for a in r['states']:self.assertEqual(a['occupancy'],pack(incoming[a['state_id']]))
        self.assertEqual(F(*r['bayes_regret']),F(*r['optimal_bayes_regret'])+F(*r['policy_loss']))
        self.assertEqual(sum(F(*a['expected_reward_contribution']) for a in r['states']),F(*r['expected_reward']))

    def test_named_demo_exact_losses_occupancies_and_localization(self):
        from banditlab.audit_demo import demo
        p=demo()['policies'];self.assertEqual(p['optimal']['expected_pulls_by_arm'],[[11,6],[7,6]]);self.assertEqual(p['greedy']['expected_pulls_by_arm'],[[2,3],[7,3]])
        self.assertEqual(p['optimal']['counts']['positive_mass_states'],14);self.assertEqual(p['fixed']['counts']['positive_mass_states'],10)
        self.assertEqual([t['weighted_optimal_bellman_gap'] for t in p['greedy']['per_pull']],[[1,36],[0,1],[0,1]])
        self.assertEqual([t['weighted_optimal_bellman_gap'] for t in p['fixed']['per_pull']],[[1,36],[8,297],[1,99]])

    def test_zero_mass_states_keep_conditional_values_not_false_visits(self):
        r=audit_policy([[1,1],[5,4]],horizon=3,policy='fixed');z=[n for n in r['states'] if n['occupancy']==[0,1] and n['remaining']]
        self.assertTrue(z);self.assertTrue(any(n['conditional_policy_value']!=[0,1] for n in z));self.assertTrue(all(n['weighted_optimal_bellman_gap']==[0,1] and all(f['probability_mass']==[0,1] for f in n['outgoing']) for n in z))

    def test_zero_horizon_single_arm_and_tie_alternative(self):
        for policy in ['optimal','greedy','fixed',[[]]]:
            r=audit_policy([[1,1],[2,2]],horizon=0,policy=policy);self.assertEqual(r['expected_reward'],[0,1]);self.assertEqual(r['terminal_probability_mass'],[1,1]);self.assertEqual(r['counts']['positive_mass_states'],1);self.assertEqual(r['per_pull'],[])
        for policy in ['optimal','greedy','fixed']:
            r=audit_policy([[3,2]],horizon=8,policy=policy);self.assertEqual(r['expected_reward'],[24,5]);self.assertEqual(r['expected_pulls_by_arm'],[[8,1]]);self.assertEqual(r['policy_loss'],[0,1])
        s=solve([[1,1],[2,2]],horizon=1);kernel=[[] if not n['remaining'] else [[0,1],[1,1]] for n in s['states']];r=audit_policy(s['priors'],horizon=1,policy=kernel);self.assertEqual(r['expected_reward'],[1,2]);self.assertEqual(r['policy_loss'],[0,1])

    def test_custom_policy_strict_validation_every_state(self):
        s=solve([[1,1],[1,1]],horizon=2);valid=[[[1,2],[1,2]] if n['remaining'] else [] for n in s['states']]
        bad=[None,{},(),True,'unknown',valid[:-1]]
        for row in [None,(),[[1,1]],[[True,2],[1,2]],[[1.0,2],[1,2]],[[1,0],[1,1]],[[-1,2],[3,2]],[[1,3],[1,3]],[[1000001,1000001],[0,1]]]:
            v=copy.deepcopy(valid);v[-1]=row;bad.append(v)
        v=copy.deepcopy(valid);v[0]=[[1,1],[0,1]];bad.append(v)
        for x in bad:
            with self.assertRaises(ValueError):audit_policy(s['priors'],horizon=2,policy=x)
        r=audit_policy(s['priors'],horizon=2,policy='fixed');z=next(n['state_id'] for n in r['states'] if n['remaining'] and n['occupancy']==[0,1]);v=copy.deepcopy(valid);v[z]=[]
        with self.assertRaises(ValueError):audit_policy(s['priors'],horizon=2,policy=v)

    def test_preflight_is_before_solution_and_policy_kernel(self):
        with patch('banditlab.audit.solve',side_effect=AssertionError('solved')),patch('banditlab.audit._kernel',side_effect=AssertionError('kernel')):
            with self.assertRaisesRegex(ValueError,'preflight refused'):audit_policy([[1,1]]*3,horizon=12,state_limit=4096,policy=[])

    def test_inputs_results_and_rng_isolation(self):
        priors=[[1,1],[5,4]];s=solve(priors,horizon=3);kernel=[[[1,3],[2,3]] if n['remaining'] else [] for n in s['states']];before=copy.deepcopy((priors,kernel));rng=random.getstate();r=audit_policy(priors,horizon=3,policy=kernel);expected=copy.deepcopy(r);r['states'][-1]['posterior'][0][0]=99;r['states'][-1]['action_probabilities'][0][0]=99
        self.assertEqual((priors,kernel),before);self.assertEqual(random.getstate(),rng);self.assertEqual(audit_policy(priors,horizon=3,policy=kernel),expected)

    def test_cli_exact_api_custom_and_named(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.json';s=solve([[1,1],[2,3]],horizon=2)
            for policy in ['optimal','greedy','fixed',[[] if not n['remaining'] else [[1,4],[3,4]] for n in s['states']]]:
                path.write_text(json.dumps({'version':1,'priors':s['priors'],'horizon':2,'policy':policy}));p=subprocess.run([sys.executable,'-m','banditlab','audit',str(path)],capture_output=True,text=True,check=True);self.assertEqual(json.loads(p.stdout),audit_policy(s['priors'],horizon=2,policy=policy))
            p=subprocess.run([sys.executable,'-m','banditlab','policy-demo'],capture_output=True,text=True,check=True)
            from banditlab.audit_demo import demo
            self.assertEqual(json.loads(p.stdout),demo())

    def test_audit_cli_missing_extra_duplicate_bad_policy_no_partial(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.json'
            for raw in ['{"version":1,"priors":[[1,1]],"horizon":1}', '{"version":1,"priors":[[1,1]],"horizon":1,"policy":"x"}', '{"version":1,"priors":[[1,1]],"horizon":1,"policy":"fixed","policy":"greedy"}', '{"version":1,"priors":[[1,1]],"horizon":1,"policy":"fixed","extra":1}']:
                path.write_text(raw);p=subprocess.run([sys.executable,'-m','banditlab','audit',str(path)],capture_output=True,text=True);self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'');self.assertIn('error:',p.stderr)

if __name__=='__main__':unittest.main()
