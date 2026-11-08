from .core import solve


def demo():
    priors=[[1,1],[5,4]]
    results=[solve(priors,horizon=h) for h in [1,2,3,4,6,8]]
    assert results[0]['optimal_actions']==[1]
    assert results[1]['optimal_actions']==[0,1] and results[1]['adaptive_gain_over_greedy']==[0,1]
    assert results[2]['optimal_actions']==[0] and results[2]['greedy_first_action']==1
    assert results[2]['value']==[187,108] and results[2]['greedy_value']==[46,27]
    assert results[2]['adaptive_gain_over_greedy']==[1,36]
    assert results[-1]['value']==[525727,108108]
    return {'format':'banditlab-demo-v1','runs':results,'selection':'Prespecified Beta(1,1) exploratory arm versus Beta(5,4) higher-prior-mean arm; horizons1,2,3,4,6,8. Exact prior expectations, not simulation trials, tuning results, empirical confidence intervals or a universal greedy-performance ranking.'}
