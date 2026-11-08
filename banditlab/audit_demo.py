from .audit import audit_policy


def demo():
    priors=[[1,1],[5,4]]
    out={p:audit_policy(priors,horizon=3,policy=p) for p in ['optimal','greedy','fixed']}
    assert out['optimal']['expected_reward']==[187,108]
    assert out['greedy']['expected_reward']==[46,27] and out['greedy']['policy_loss']==[1,36]
    assert out['fixed']['expected_reward']==[5,3] and out['fixed']['policy_loss']==[7,108]
    return {'format':'banditlab-policy-demo-v1','policies':out,'selection':'Prescribed three-pull Beta(1,1)/Beta(5,4) example. Exact execution flows under each fully specified policy, not experiments or a universal greedy ranking.'}
