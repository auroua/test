__author__ = 'auroua'
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

data  = np.loadtxt('txtdata.csv')
alpha = 1.0/data.mean()

lambda_1 = pm.Exponential('lambda_1',alpha)
lambda_2 = pm.Exponential('lambda_2',alpha)
tau = pm.DiscreteUniform('tau',lower=0,upper=len(data))

print 'random numbers',tau.random(),tau.random()

@pm.deterministic
def lambda_(t=tau,l_1=lambda_1,l_2=lambda_2):
    out = np.ones(len(data),dtype=int)
    out[:tau] = l_1
    out[tau:] = l_2
    return out

obversations = pm.Poisson('obs',lambda_,value=data,observed=True)

model = pm.Model([obversations,tau,lambda_1,lambda_2])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)

lambda_1_samples = mcmc.trace('lambda_1')[:]
lambda_2_samples = mcmc.trace('lambda_2')[:]
tau_samples = mcmc.trace('tau')[:]

print tau_samples

# histogram of the samples:

ax = plt.subplot(411)
ax.set_autoscaley_on(False)

plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_1$", color="#A60628", normed=True)
plt.legend(loc="upper left")
plt.title(r"""Posterior distributions of the variables
    $\lambda_1,\;\lambda_2,\;\tau$""")
plt.xlim([15, 30])
plt.xlabel("$\lambda_1$ value")

ax = plt.subplot(412)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_2$", color="#7A68A6", normed=True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel("$\lambda_2$ value")

plt.subplot(413)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=len(data), alpha=1,
         label=r"posterior of $\tau$",
         color="#467821", weights=w, rwidth=2.)
plt.xticks(np.arange(len(data)))

plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(data) - 20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel("probability")


# tau_samples, lambda_1_samples, lambda_2_samples contain
# N samples from the corresponding posterior distribution
N = tau_samples.shape[0]
print N
expected_texts_per_day = np.zeros(len(data))
for day in range(0, len(data)):
    # ix is a bool index of all tau samples corresponding to
    # the switchpoint occurring prior to value of 'day'
    ix = day < tau_samples
    # Each posterior sample corresponds to a value for tau.
    # for each day, that value of tau indicates whether we're "before"
    # (in the lambda1 "regime") or
    #  "after" (in the lambda2 "regime") the switchpoint.
    # by taking the posterior sample of lambda1/2 accordingly, we can average
    # over all samples to get an expected value for lambda on that day.
    # As explained, the "message count" random variable is Poisson distributed,
    # and therefore lambda (the poisson parameter) is the expected value of
    # "message count".
    expected_texts_per_day[day] = (lambda_1_samples[ix].sum()
                                   + lambda_2_samples[~ix].sum()) / N


plt.subplot(414)
plt.plot(range(len(data)), expected_texts_per_day, lw=4, color="#E24A33",
         label="expected number of text-messages received")
plt.xlim(0, len(data))
plt.xlabel("Day")
plt.ylabel("Expected # text-messages")
plt.title("Expected number of text-messages received")
plt.ylim(0, 60)
plt.bar(np.arange(len(data)), data, color="#348ABD", alpha=0.65,
        label="observed texts per day")

plt.legend(loc="upper left")

plt.show()


#exercise 1
print lambda_1_samples.mean(),lambda_2_samples.mean()

#exercise 2
rates = lambda_1_samples/lambda_2_samples
print rates
print rates.mean()
print 'ex2------------------'
print lambda_1_samples.mean()/lambda_2_samples.mean()

print 'ex3-------------------'
#exercise 3
ix_3 = tau_samples<45
print np.sum(ix_3)
final_lambda = lambda_1_samples[ix_3].sum()/np.sum(ix_3)
print final_lambda

