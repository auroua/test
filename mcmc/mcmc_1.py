#encoding : UTF-8
__author__ = 'auroua'
import random

#beta distribution
def beta_s(w,a,b):
    return w**(a-1)*(1-w)**(b-1)

def random_coin(p):
    unif = random.uniform(0,1)
    if unif>=p:
        return False
    else:
        return True

def beta_mcmc(N_hops,a,b):
    states = []
    cur = random.uniform(0,1)
    for i in range(0,N_hops):
        states.append(cur)
        next = random.uniform(0,1)
        ap = min(beta_s(next,a,b)/beta_s(cur,a,b,),1)
        if random_coin(ap):
            cur = next
    return states[-700000:]