import torch
from pml.gmm_em import gmm_em
'''
Code for inference and learning in HMMs with gaussian observations

As explained in "Machine Learning - A Probabilistic Perspective" page 609ff, 619ff
Notation deviates from those pages, but is chosen to align better with other pages on kalman filters, switching KF

# pi -- the prior belief over the individual domains
# p -- the online belief over the individual domains
# z -- the observation
# A -- the state transition probabilities for discrete components
# mu -- the mean observation per component 
# Sigma -- the covariances of observation per component
'''

def create_hmm_data(pi, A, B, R):
    # create input data
    T = 120  # number of time-steps
    ss = torch.zeros(T, dtype=torch.long)  # trajectory of discrete states
    zs = torch.zeros(T, dim_observation)  # trajectory of observations

    # sample for t=0 (special case, no previous time-step)
    categorical = torch.distributions.Categorical(pi)
    s = categorical.sample()
    for t in range(1, T):
        # sample the discrete component
        categorical = torch.distributions.Categorical(A[ss[t - 1]])
        s = categorical.sample((1,))
        ss[t] = s

        # sample the observation
        distrib = torch.distributions.MultivariateNormal(loc=B[s], covariance_matrix=R[s])
        zs[t] = distrib.sample()
    return zs, ss

def predict_hmm(p, A):
    p_bar = A.mv(p)
    p_bar = p_bar / torch.sum(p_bar)
    return p_bar
def update_hmm(p, z, B, R, is_return_likelihoods=False):
    # compute measurement likelihood
    z = z.view(1,-1)
    exp = torch.exp(-0.5*torch.einsum('bi,bij,bj->b', (z - B), R, (z - B)))
    likelihood = torch.det(R)**-.5 * exp # isn't normalized properly up to constant, but doesn't matter

    if torch.var(likelihood) == 0:
        likelihood = torch.ones_like(likelihood) / likelihood.numel()  # crappy fix for all zero likelihoods

    p_hat = (p * likelihood)
    p_hat = p_hat / torch.sum(p_hat)
    if is_return_likelihoods:
        return p_hat, likelihood
    else:
        return p_hat

def forward_backward_hmm(pi, zs, A, B, R, is_return_likelihoods=False):
    n_components = pi.numel()
    T, dim_observation = zs.size()

    ps = torch.zeros(T,n_components) # the discrete belief for each time-step
    p = pi
    likelihoods = torch.zeros(T,n_components) # for each timestep, the likelihood that each state generate the observation

    # forward pass
    for i in range(T):
        p_bar = predict_hmm(p, A)
        ps[i], likelihoods[i] = update_hmm(p_bar, zs[i], B, R, is_return_likelihoods=True)
        p = ps[i]

    # backward pass
    beta = torch.ones(T, n_components)
    for i in range(T-2, -1, -1):
        beta[i] = A.mv(likelihoods[i+1]*beta[i+1]) # probably need to fix transition backwards
        # beta[i] = torch.matmul(likelihoods[i + 1] * beta[i + 1], A.inverse())  # probably need to fix transition backwards
        beta[i] /= torch.sum(beta[i])

    ps_bar = ps * beta
    ps = ps_bar / ps_bar.sum(dim=1).unsqueeze(1)
    if is_return_likelihoods:
        return ps, likelihoods
    else:
        return ps

def em_hmm(zs, K, n_iter=10, initialization="gmm"):
    """
    estimates parameters for a hmm with gaussian observations using EM
    Args:
        zs (torch.Tensor): the observations
        K (torch.Tensor): the number of components
        n_iter (torch.Tensor): the number of iterations to do EM
        initialization (torch.Tensor): the type of initialization. allowed values: "random", "gmm", or a tuple (pi, A, B, R)
    Returns:
        pi, A, B, R: the parameters of the HMM
    """
    dim_observation = 2

    if type(initialization) == tuple:
        pi, A, B, R = initialization
    elif initialization == "gmm":
        B, R = gmm_em(zs, K, n_epochs=100) # @todo get pi out of this

        print("GMM initialized B and R")
        print(B)
        print(R)

        A = torch.normal(torch.zeros(K,K), torch.ones(K,K)) # state transition matrix
        pi = torch.tensor([.5, .5])

    elif initialization == "random":
        # sample random A and B
        A = torch.normal(torch.zeros(K,K), torch.ones(K,K)) # state transition matrix
        B = torch.normal(torch.zeros(K,dim_observation), torch.ones(K,dim_observation))
        # initialize Q, pi somehow
        R = torch.eye(dim_observation).unsqueeze(0).repeat(K,1,1)
        pi = torch.tensor([.5, .5])
    else:
        raise Exception("Unknown initialization")

    for iteration in range(n_iter):
        # Expect
        ps, likelihood = forward_backward_hmm(pi, zs, A, B, R, is_return_likelihoods=True)

        # Maximize
        pi = ps[0]
        for k in range(K):
            # compute state transition matrix
            expected = torch.sum(ps[:-1,k].unsqueeze(1) * ps [1:,:], dim=0)
            A[k,:] = expected / torch.sum(expected)

            # compute gaussian parameters
            B[k] = torch.sum(ps[:,k].unsqueeze(1) * zs, dim=0) / torch.sum(ps[:,k])
    return pi, A, B, R

if __name__ == "__main__":
    #torch.manual_seed(1)
    dim_observation = 2
    pi = torch.tensor([.5, .5])
    A = torch.tensor([[.9, .1],
                      [.1, .9]])

    B = torch.tensor([[0., 4.1],
                      [0., .9]])
    R = torch.tensor([[[1., 0.],
                       [0., 1.]],
                      [[1., 0.],
                       [0., 1.]]])

    zs, ss = create_hmm_data(pi, A, B, R)
    print("ground truth states:", ss)
    T = zs.size()[0]
    # inference with known model
    ps = torch.zeros(T,2)
    p = pi
    for i in range(0,T):
        p = update_hmm(p, zs[i], B, R)
        ps[i] = p
        p = predict_hmm(p, A)


    torch.set_printoptions(precision=2)
    print("state predictions forward:", ps[:,1].round().int())
    # print("forward:", ps[:,1])
    ps = forward_backward_hmm(pi, zs, A, B, R)
    print("state predictions forback:", ps[:,1].round().int())


    # pi, A, B, R = em_hmm(zs, K=2, n_iter=30, initialization=(pi, A, B, R))
    # pi, A, B, R = em_hmm(zs, K=2, n_iter=30, initialization="random")
    pi_emm, A_emm, B_emm, R_emm = em_hmm(zs, K=2, n_iter=30, initialization="gmm")

    ps = forward_backward_hmm(pi_emm, zs, A_emm, B_emm, R_emm)
    print("state predictions learned:", ps[:,1].round().int())
    # print("learned:", ps[:,1])