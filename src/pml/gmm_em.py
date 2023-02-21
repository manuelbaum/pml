import torch
import matplotlib.pyplot as plt

"""
This script implements expectation maximization for gaussian mixture models.
It follows the description in 'Probabilistic Machine Learning' by Kevin Murphy.
Page 352, Chapter 11.4.2 EM for GMMs
"""

def create_gmm_data() -> torch.Tensor:
    '''
    Creates 2d training data for a gaussian mixture model by sampling from 3 different gaussian components
    Returns:
        torch.Tensor: A tensor of size n_data x 2
    '''

    n_data_per_class = torch.tensor([400, 300, 200])
    n_components = n_data_per_class.numel()
    mu = torch.tensor([[-5., 0],# the mean for each of the three components
                       [0., 4.],
                       [3., 3.]])
    Sigma = torch.tensor([[[1., 0.], # the covariance for each of the three components
                           [0., 1.]],
                          [[10., 0.],
                           [0., 1.]],
                          [[1., 0.],
                           [0., 1.]]])

    xs = torch.empty([0, 2])
    for i_component in range(n_components):
        mvn = torch.distributions.MultivariateNormal(mu[i_component], Sigma[i_component])
        xs = torch.cat((xs, mvn.sample(n_data_per_class[i_component].unsqueeze(0))))
    return xs

def compute_gmm_likelihood(xs, mu, Sigma):
    """
    Computes the likelihood that a batch of input data was generated from one out of several gaussian components
    Args:
        xs (torch.Tensor): input data of shape n_data x dim_data
        mu (torch.Tensor): means of the components, shape n_components x dim_data
        Sigma (torch.Tensor): covariance of the components, shape n_components x dim_data x dim_data

    Returns:
        likelihood (torch.Tensor): the per-data-point likelihood that the data was generated from each component. shape: n_data x n_components
    """
    n_data = xs.size()[0]
    n_components = mu.size()[0]
    likelihood = torch.zeros(n_data, n_components)
    for i_component in range(n_components):
        S = Sigma[i_component]
        nu = xs - mu[i_component]
        mahalanobis = torch.einsum("bi,ij,bj->b",nu,torch.inverse(Sigma[i_component]),nu)
        likelihood[:, i_component] = torch.det(S) ** -.5 * torch.exp(-.5 * mahalanobis)
    return likelihood

def gmm_em(xs, n_components, n_epochs = 100):
    '''
    This function fits a Gaussian Mixture Model to the input data. The implementation is not fully parallelized, it loops
    over the components in several places. This would be rather easy to optimize in the future and for me the trade-off
    is not worth it right now.
    Args:
        xs (torch.Tensor): the input data, shape n_data x dim_data
        n_components (int): the number of hidden components
        n_epochs (int): the number of iterations to perform

    Returns:
    mu (torch.Tensor): the means for each component, shape n_components x dim_data
    Sigma (torch.Tensor): the covariance for each components, shape n_components x dim_data x dim_data
    '''
    n_data, dim_data = xs.size()

    # create initial parameters
    pi = torch.ones(n_components)/n_components  # uniform distribution
    # fit gaussian to all of data, then sample per component means from that and over-estimate cov to be shared cov
    mu_all = torch.mean(xs, dim=0)
    Sigma_all = torch.cov(xs.T)
    mvn = torch.distributions.MultivariateNormal(mu_all, Sigma_all)
    mu = mvn.sample([n_components]) # sample new means
    Sigma = Sigma_all.unsqueeze(0).repeat(n_components,1,1) # reuse (over-estimate) the covariance for each component

    # precompute outer product for each individual datapoint
    xxt = torch.einsum("bi,bj->bij",xs,xs)

    # do em to fit the gmm model
    for i_epoch in range(n_epochs):

        ### expect: compute new responsibilty of components for data points, given fixed parameters for distributions
        likelihood = compute_gmm_likelihood(xs, mu, Sigma)
        r = pi.unsqueeze(0) * likelihood # r is the responsibility of components for data-points
        r /= r.sum(dim=1).unsqueeze(1)
        r_marginal = torch.sum(r, dim=0)

        ### maximize
        pi = r_marginal /n_data # update prior probability for components
        for i_component in range(n_components):
            mu[i_component] = torch.mv(xs.T,r[:,i_component]) / r_marginal[i_component]
            Sigma[i_component] = torch.einsum("bij,b->ij", xxt, r[:,i_component])/r_marginal[i_component] - mu[i_component].outer(mu[i_component])

    return mu, Sigma




def main():
    xs = create_gmm_data()

    # fit

    n_components = 3
    mu, Sigma  = gmm_em(xs, n_components)

    # visualize

    likelihood = compute_gmm_likelihood(xs, mu, Sigma)
    categories = torch.argmax(likelihood,dim=1)

    cmap = ["r","g","b"]
    plt.scatter(*xs.T, c=[cmap[c] for c in categories])
    plt.scatter(*mu.T, c = 'k', marker="x")
    plt.show()

if __name__ == "__main__":
    main()