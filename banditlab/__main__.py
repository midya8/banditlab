import argparse,json
from .core import solve
from .demo import demo


def pairs(items):
    out={}
    for k,v in items:
        if k in out:raise ValueError('duplicate JSON key: '+k)
        out[k]=v
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description='Exact Beta-Bernoulli finite-horizon bandit planning');s=p.add_subparsers(dest='command',required=True);s.add_parser('demo');a=s.add_parser('solve');a.add_argument('input');a.add_argument('--state-limit',type=int,default=4096);args=p.parse_args(argv)
    try:
        if args.command=='demo':out=demo()
        else:
            with open(args.input,'rb') as f:raw=f.read(65537)
            if len(raw)>65536:raise ValueError('input JSON exceeds 64 KiB')
            x=json.loads(raw,object_pairs_hook=pairs)
            if type(x) is not dict or set(x)!=set(['version','priors','horizon']) or type(x['version']) is not int or x['version']!=1:raise ValueError('input requires exactly version=1, priors and horizon')
            out=solve(x['priors'],horizon=x['horizon'],state_limit=args.state_limit)
        print(json.dumps(out,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError,OverflowError) as e:p.exit(2,'error: '+str(e)+'\n')

if __name__=='__main__':main()
