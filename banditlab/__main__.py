import argparse,json
from .core import solve
from .demo import demo
from .audit import audit_policy
from .audit_demo import demo as audit_demo


def pairs(items):
    out={}
    for k,v in items:
        if k in out:raise ValueError('duplicate JSON key: '+k)
        out[k]=v
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description='Exact Beta-Bernoulli finite-horizon bandit planning');s=p.add_subparsers(dest='command',required=True);s.add_parser('demo');s.add_parser('policy-demo')
    for command in ['solve','audit']:
        a=s.add_parser(command);a.add_argument('input');a.add_argument('--state-limit',type=int,default=4096)
    args=p.parse_args(argv)
    try:
        if args.command=='demo':out=demo()
        elif args.command=='policy-demo':out=audit_demo()
        else:
            with open(args.input,'rb') as f:raw=f.read(65537)
            if len(raw)>65536:raise ValueError('input JSON exceeds 64 KiB')
            x=json.loads(raw,object_pairs_hook=pairs)
            keys=['version','priors','horizon']+(['policy'] if args.command=='audit' else [])
            if type(x) is not dict or set(x)!=set(keys) or type(x['version']) is not int or x['version']!=1:raise ValueError('input requires version=1, priors, horizon and policy for audit, with no extra keys')
            out=(audit_policy(x['priors'],horizon=x['horizon'],policy=x['policy'],state_limit=args.state_limit) if args.command=='audit' else solve(x['priors'],horizon=x['horizon'],state_limit=args.state_limit))
        print(json.dumps(out,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError,OverflowError) as e:p.exit(2,'error: '+str(e)+'\n')

if __name__=='__main__':main()
