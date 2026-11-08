"""Exact forward execution flows and finite-horizon performance differences."""
from fractions import Fraction as F
from .core import solve,pack,validate


def _kernel(policy,solution):
    states=solution['states'];k=len(solution['priors'])
    if type(policy) is str:
        if policy not in ('optimal','greedy','fixed'):raise ValueError('named policy must be optimal, greedy or fixed')
        fixed=solution['states'][-1]['myopic_maximizers'][0]
        out=[]
        for n in states:
            if not n['remaining']:out.append([]);continue
            a=n['selected_action'] if policy=='optimal' else n['greedy_action'] if policy=='greedy' else fixed
            out.append([F(int(i==a)) for i in range(k)])
        return out,policy
    if type(policy) is not list or len(policy)!=len(states):raise ValueError('custom policy needs one probability row per returned state ID')
    out=[]
    for n,row in zip(states,policy):
        if type(row) is not list or len(row)!=(k if n['remaining'] else 0):raise ValueError('decision rows need one rational per arm; terminal rows must be empty')
        vals=[]
        for pair in row:
            if type(pair) is not list or len(pair)!=2 or any(type(v) is not int for v in pair) or not 0<=pair[0]<=1000000 or not 1<=pair[1]<=1000000:raise ValueError('probabilities need integer [numerator,denominator], 0<=n<=1000000 and 1<=d<=1000000')
            vals.append(F(*pair))
        if n['remaining'] and sum(vals)!=1:raise ValueError('action probabilities must sum to exactly one in every decision state')
        out.append(vals)
    return out,'custom-state-randomized'


def audit_policy(priors,*,horizon=4,policy='greedy',state_limit=4096):
    """Audit a named policy or a complete state-indexed exact probability kernel.

    Custom rows follow solve(...)["states"] IDs, with [] at terminal states.
    Random draws are independent conditional on the current information state.
    All decision states are required, even when their forward occupancy is zero.
    """
    validate(priors,horizon,state_limit)
    s=solve(priors,horizon=horizon,state_limit=state_limit);states=s['states'];k=len(priors);kernel,name=_kernel(policy,s)
    values=[];gaps=[]
    for n,probs in zip(states,kernel):
        value=F(0)
        for a,pa in enumerate(probs):
            for reward,key in [(0,'failure'),(1,'success')]:
                b=n['branches'][a][key];value+=pa*F(*b['probability'])*(reward+values[b['state_id']])
        values.append(value)
        gaps.append(F(*n['value'])-sum(pa*F(*n['action_values'][a]) for a,pa in enumerate(probs)))
        assert gaps[-1]>=0
    mass=[F(0)]*len(states);mass[s['root_state_id']]=F(1);flows=[[] for _ in states]
    steps=[{'pull':t+1,'arms':[F(0)]*k,'successes':[F(0)]*k,'failures':[F(0)]*k,'reward':F(0),'loss':F(0),'mass':F(0),'visited':0} for t in range(horizon)]
    for n in reversed(states):
        sid=n['id'];occupancy=mass[sid]
        if not n['remaining']:continue
        step=steps[n['elapsed']];step['mass']+=occupancy;step['visited']+=int(occupancy>0);step['loss']+=occupancy*gaps[sid]
        for a,pa in enumerate(kernel[sid]):
            step['arms'][a]+=occupancy*pa
            for reward,key in [(0,'failure'),(1,'success')]:
                b=n['branches'][a][key];flow=occupancy*pa*F(*b['probability']);mass[b['state_id']]+=flow
                step['successes' if reward else 'failures'][a]+=flow;step['reward']+=reward*flow
                flows[sid].append({'action':a,'reward':reward,'child_state_id':b['state_id'],'probability_mass':pack(flow)})
    rows=[]
    for n in states:
        sid=n['id'];reward=mass[sid]*sum(pa*F(*n['posterior_means'][a]) for a,pa in enumerate(kernel[sid]))
        rows.append({'state_id':sid,'elapsed':n['elapsed'],'remaining':n['remaining'],'posterior':n['posterior'],'occupancy':pack(mass[sid]),'action_probabilities':list(map(pack,kernel[sid])),
                     'conditional_policy_value':pack(values[sid]),'expected_reward_contribution':pack(reward),'optimal_bellman_gap':pack(gaps[sid]),'weighted_optimal_bellman_gap':pack(mass[sid]*gaps[sid]),'outgoing':flows[sid]})
    total=sum(t['reward'] for t in steps);loss=sum(t['loss'] for t in steps);optimal=F(*s['value']);terminal=sum(mass[n['id']] for n in states if not n['remaining'])
    assert terminal==1 and total==values[-1] and loss==optimal-total
    per_pull=[]
    for t in steps:
        assert t['mass']==1 and sum(t['arms'])==1 and sum(t['successes'])==t['reward'] and sum(t['successes'])+sum(t['failures'])==1
        per_pull.append({'pull':t['pull'],'state_mass':pack(t['mass']),'positive_mass_states':t['visited'],'arm_probability_mass':list(map(pack,t['arms'])),'success_mass_by_arm':list(map(pack,t['successes'])),'failure_mass_by_arm':list(map(pack,t['failures'])),'expected_reward':pack(t['reward']),'weighted_optimal_bellman_gap':pack(t['loss'])})
    upper=F(*s['clairvoyant']['total_expected_reward_upper_bound'])
    return {'format':'banditlab-policy-audit-v1','policy':name,'priors':[p[:] for p in priors],'horizon':horizon,'root_state_id':s['root_state_id'],
            'expected_reward':pack(total),'optimal_value':s['value'],'policy_loss':pack(loss),'weighted_bellman_gap_sum':pack(loss),'clairvoyant_upper_bound':pack(upper),
            'bayes_regret':pack(upper-total),'optimal_bayes_regret':s['clairvoyant']['optimal_bayes_regret'],'terminal_probability_mass':pack(terminal),
            'expected_pulls_by_arm':[pack(sum(t['arms'][a] for t in steps)) for a in range(k)],'per_pull':per_pull,'states':rows,
            'counts':{'full_graph_states':len(states),'positive_mass_states':sum(m>0 for m in mass),'state_limit':state_limit},
            'scope':'exact supplied-prior finite-horizon execution, not sampled or empirical behavior; complete probability kernel on information states, independent randomization conditional on that state; first-index ties in named optimal/greedy/fixed; fixed commits to the best initial mean arm; occupancy merges history probabilities, not counts of histories; zero-mass states retain conditional policy values but contribute zero loss; local gap uses optimal continuation Q, not the policy continuation Q; total weighted gap equals optimal value minus policy value; clairvoyant knows latent rates, not reward draws; decomposition is not a causal or frequentist regret guarantee'}
