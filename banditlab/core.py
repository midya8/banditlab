"""Exact finite-horizon Bayes-adaptive planning for independent Beta arms."""
from fractions import Fraction as F
from math import comb,factorial


def pack(x):return [x.numerator,x.denominator]
def validate(priors,horizon,state_limit):
    if type(priors) is not list or not 1<=len(priors)<=3:raise ValueError('priors must be a list of 1..3 [alpha,beta] pairs')
    if any(type(p) is not list or len(p)!=2 or any(type(x) is not int or not 1<=x<=16 for x in p) for p in priors):raise ValueError('prior alpha/beta must be integers in 1..16, not booleans')
    if type(horizon) is not int or not 0<=horizon<=12:raise ValueError('horizon must be integer in 0..12')
    if type(state_limit) is not int or not 1<=state_limit<=20000:raise ValueError('state_limit must be integer in 1..20000')
    planned=comb(horizon+2*len(priors),2*len(priors))
    if planned>state_limit:raise ValueError('state preflight refused: '+str(planned)+' posterior/time states exceeds limit '+str(state_limit)+'; no partial result')
    return planned


def _compositions(total,width):
    if width==1:
        yield (total,);return
    for n in range(total+1):
        for rest in _compositions(total-n,width-1):yield (n,)+rest


def _cdf(a,b):
    # Integral of x**(a-1)*(1-x)**(b-1)/B(a,b), using the binomial expansion.
    norm=factorial(a+b-1)//(factorial(a-1)*factorial(b-1));out=[F(0)]*(a+b)
    for j in range(b):out[a+j]=F(norm*(-1)**j*comb(b-1,j),a+j)
    return out


def _clairvoyant_mean(priors):
    product=[F(1)]
    for a,b in priors:
        cdf=_cdf(a,b);out=[F(0)]*(len(product)+len(cdf)-1)
        for i,x in enumerate(product):
            for j,y in enumerate(cdf):out[i+j]+=x*y
        product=out
    # E max(theta_i) = integral_0^1 (1 - product_i F_i(x)) dx.
    mean=1-sum(c/F(j+1) for j,c in enumerate(product))
    assert 0<=mean<=1
    return mean,product


def solve(priors,*,horizon=4,state_limit=4096):
    planned=validate(priors,horizon,state_limit);k=len(priors);table=[];lookup={};edges=0
    # All weak compositions of elapsed pulls over 2*K success/failure counts.
    # Each state is reachable under some action/outcome history, not necessarily the selected policy.
    for elapsed in range(horizon,-1,-1):
        left=horizon-elapsed
        for inc in _compositions(elapsed,2*k):
            posterior=[[priors[a][0]+inc[2*a],priors[a][1]+inc[2*a+1]] for a in range(k)]
            means=[F(a,a+b) for a,b in posterior];best_mean=max(means);myopic_ties=[a for a,m in enumerate(means) if m==best_mean];greedy_action=myopic_ties[0] if left else None
            q=[];gq=[];branches=[]
            if left:
                for a,mean in enumerate(means):
                    children=[]
                    for reward in [0,1]:
                        nxt=list(inc);nxt[2*a+(0 if reward else 1)]+=1;cid=lookup[tuple(nxt)];children.append(table[cid])
                    fail,success=children
                    q.append(mean*(1+F(*success['value']))+(1-mean)*F(*fail['value']))
                    gq.append(mean*(1+F(*success['greedy_value']))+(1-mean)*F(*fail['greedy_value']))
                    branches.append({'action':a,'failure':{'probability':pack(1-mean),'state_id':fail['id']},'success':{'probability':pack(mean),'state_id':success['id']}});edges+=2
                value=max(q);ties=[a for a,v in enumerate(q) if v==value];gv=gq[greedy_action]
            else:value=F(0);gv=F(0);ties=[]
            node={'id':len(table),'elapsed':elapsed,'remaining':left,'increments':list(inc),'posterior':posterior,
                  'posterior_means':list(map(pack,means)),'action_values':list(map(pack,q)),
                  'value':pack(value),'optimal_actions':ties,'selected_action':ties[0] if ties else None,
                  'myopic_maximizers':myopic_ties,'greedy_action':greedy_action,'greedy_continuation_action_values':list(map(pack,gq)),
                  'greedy_value':pack(gv),'best_fixed_arm_value':pack(left*best_mean),'branches':branches}
            assert value>=gv and value>=left*best_mean
            lookup[inc]=node['id'];table.append(node)
    assert len(table)==planned
    root=table[-1];oracle_mean,cdf=_clairvoyant_mean(priors);upper=horizon*oracle_mean;value=F(*root['value']);greedy=F(*root['greedy_value']);fixed=F(*root['best_fixed_arm_value'])
    assert upper>=value
    return {'format':'banditlab-solution-v1','priors':[p[:] for p in priors],'horizon':horizon,'root_state_id':root['id'],
            'value':pack(value),'action_values':root['action_values'][:],'optimal_actions':root['optimal_actions'][:],
            'selected_action':root['selected_action'],'greedy_value':pack(greedy),'greedy_first_action':root['greedy_action'],
            'best_fixed_arm_value':pack(fixed),'best_fixed_arms':root['myopic_maximizers'][:] if horizon else [],
            'adaptive_gain_over_greedy':pack(value-greedy),'adaptive_gain_over_fixed':pack(value-fixed),
            'clairvoyant':{'mean_best_parameter':pack(oracle_mean),'total_expected_reward_upper_bound':pack(upper),
                           'max_parameter_cdf_polynomial':list(map(pack,cdf)),'optimal_bayes_regret':pack(upper-value),
                           'greedy_bayes_regret':pack(upper-greedy),'fixed_bayes_regret':pack(upper-fixed)},
            'counts':{'preflight_states':planned,'returned_states':len(table),'terminal_states':comb(horizon+2*k-1,2*k-1),
                      'decision_states':len(table)-comb(horizon+2*k-1,2*k-1),'action_outcome_edges':edges,'state_limit':state_limit},
            'states':table,
            'scope':'independent fixed latent arm rates drawn once from supplied integer Beta priors; conditionally IID unit Bernoulli rewards; exact finite-horizon undiscounted prior expectation, not a fixed-environment frequentist performance guarantee; non-anticipating adaptive optimum; greedy updates posteriors after EVERY outcome under the SAME horizon; fixed-arm commitment uses best prior mean; clairvoyant knows latent rates, not future reward draws, and supplies an information-advantaged comparison upper bound; state table includes all states reachable under ANY actions, not only selected-policy occupancy; first-index tie selection; count cap does not bound Fraction bits, wall time or RAM; no external data, online service, nonstationarity or general bandit optimality claim'}
