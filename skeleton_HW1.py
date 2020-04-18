#Filename: HW1_skeleton.py

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

from numpy.linalg import inv, norm  #for GN method

#--------------------------------------------------------------------------------
plot_converging = False

# Assignment 1
def main():
    for i in range(0,3):
        task(i+1)

def task(scenario):
    
    # choose the scenario
    #scenario = 2    # all anchors are Gaussian
    #scenario = 2    # 1 anchor is exponential, 3 are Gaussian
    #scenario = 3    # all anchors are exponential
    
    # specify position of anchors
    p_anchor = np.array([[5,5],[-5,5],[-5,-5],[5,-5]])
    nr_anchors = np.size(p_anchor,0)
    
    # position of the agent for the reference mearsurement
    p_ref = np.array([[0,0]])
    # true position of the agent (has to be estimated)
    p_true = np.array([[2,-4]])
                       
    # plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref)
    
    # load measured data and reference measurements for the chosen scenario
    data,reference_measurement = load_data(scenario)
    
    # get the number of measurements 
    assert(np.size(data,0) == np.size(reference_measurement,0))
    nr_samples = np.size(data,0)
    
    #1) ML estimation of model parameters
    #TODO 
    #params = parameter_estimation(reference_measurement,nr_anchors,p_anchor,p_ref)
    
    #2) Position estimation using least squares
    pad_far = 5
    pad_near = 0.5
    if (scenario == 1):
        position_estimation_least_squares(data, nr_anchors, p_anchor, p_true, True, "Position estimation, scenario "+str(scenario),pad=pad_far)
        plot_anchors_and_agent(nr_anchors, p_anchor, p_true)
        #plt.show()
        plt.savefig('sc1_gn_far.svg')

        position_estimation_least_squares(data, nr_anchors, p_anchor, p_true, True, "Position estimation, scenario "+str(scenario),pad=pad_near)
        #plt.show()
        plt.savefig('sc1_gn_near.svg')

    if(scenario == 2):
        position_estimation_least_squares(data, nr_anchors, p_anchor, p_true, False, "Position estimation, scenario "+str(scenario)+" without exp. anchor",pad=pad_far)
        plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref, use_exp=False)
        plt.savefig('sc2_gn_far_wo_exp.svg')
        position_estimation_least_squares(data, nr_anchors, p_anchor, p_true, False, "Position estimation, scenario "+str(scenario)+" without exp. anchor",pad=pad_near)
        plt.savefig('sc2_gn_near_wo_exp.svg')


        position_estimation_least_squares(data, nr_anchors, p_anchor, p_true, True, "Position estimation, scenario "+str(scenario)+" with exp. anchor",pad=pad_far)
        plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref, use_exp=True)
        plt.savefig('sc2_gn_far_w_exp.svg')
        position_estimation_least_squares(data, nr_anchors, p_anchor, p_true, True, "Position estimation, scenario "+str(scenario)+" with exp. anchor",pad=pad_near)
        plt.savefig('sc2_gn_near_w_exp.svg')
    if (scenario == 3):
        position_estimation_least_squares(data, nr_anchors, p_anchor, p_true, True, "Position estimation, scenario "+str(scenario),pad=pad_far)
        plot_anchors_and_agent(nr_anchors, p_anchor, p_true)
        plt.savefig('sc3_gn_far.svg')
        position_estimation_least_squares(data, nr_anchors, p_anchor, p_true, True, "Position estimation, scenario "+str(scenario),pad=pad_near)
        plt.savefig('sc3_gn_near.svg')

    if(scenario == 3):
        # TODO: don't forget to plot joint-likelihood function for the first measurement

        #3) Postion estimation using numerical maximum likelihood
        #TODO
        #position_estimation_numerical_ml(data,nr_anchors,p_anchor, params, p_true)
    
        #4) Position estimation with prior knowledge (we roughly know where to expect the agent)
        #TODO
        # specify the prior distribution
        prior_mean = p_true
        prior_cov = np.eye(2)
        #position_estimation_bayes(data,nr_anchors,p_anchor,prior_mean,prior_cov, params, p_true)

    pass

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def parameter_estimation(reference_measurement,nr_anchors,p_anchor,p_ref):
    """ estimate the model parameters for all 4 anchors based
     on the reference measurements, i.e., for anchor i consider reference_measurement[:,i]
    Input:
        reference_measurement... nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2
        p_ref... reference point, 2x2 """

    params = np.zeros([3, nr_anchors])

    colors = ['red', 'orange', 'green', 'blue']
    num_bins = 120
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(reference_measurement[:,0], num_bins, facecolor=colors[0], alpha=0.40)
    axs[0, 0].set_title('Anchor 1')
    axs[0, 1].hist(reference_measurement[:,1], num_bins, facecolor=colors[1], alpha=0.40)
    axs[0, 1].set_title('Anchor 2')
    axs[1, 0].hist(reference_measurement[:,2], num_bins, facecolor=colors[2], alpha=0.40)
    axs[1, 0].set_title('Anchor 3')
    axs[1, 1].hist(reference_measurement[:,3], num_bins, facecolor=colors[3], alpha=0.40)
    axs[1, 1].set_title('Anchor 4')

    for ax in axs.flat:
        ax.set(xlabel='Bins', ylabel='Amount')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    for ax in fig.get_axes():
        ax.label_outer()

    plt.show()


    for i, column in enumerate(reference_measurement.T):
        # mμ
        params[0, i] = np.mean(column)

        # sigma
        params[1,i] = np.sqrt(np.mean((column - np.mean(column))**2))
        
        # lambda
        real_dist = np.linalg.norm(p_anchor[i]- p_ref)
        params[2,i] = 1 / np.mean(column - real_dist)


        print((np.subtract(column, params[0, i])))
    print(params)
    #TODO (2) estimate the according parameter based 
    return params
#--------------------------------------------------------------------------------
def position_estimation_least_squares(data,nr_anchors,p_anchor, p_true, use_exponential, title, pad=10.2):
    """estimate the position by using the least squares approximation. 
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2 
        p_true... true position (needed to calculate error) 2x2 
        use_exponential... determines if the exponential anchor in scenario 2 is used, bool"""

    tol = 1e-9  # tolerance
    max_iter = 20  # maximum iterations for GN


    fig, axs = plt.subplots(2)
    fig.suptitle('Distance from converging point, 200 measurments')

    p_ls = []
    for i, row in enumerate(data):
        random_init_pos = np.random.uniform(-5, 5, 2)
        if not use_exponential:
            ls = least_squares_GN(p_anchor, random_init_pos, row, max_iter, tol, axs)
        else:
            ls = least_squares_GN(p_anchor[1:], random_init_pos, row[1:], max_iter, tol, axs)
        p_ls.append(ls)
        if i == 20000:
            break;
    p_ls = np.array(p_ls)

    if plot_converging:
        plt.yscale("log")
        plt.xlabel("i")
        plt.ylabel("d(p(i),p(i_max)) log")
        plt.show()
    plt.clf()

    #calc mean
    N = np.size(p_ls,0)
    sum = np.array([0,0])
    for p in p_ls:
        sum = sum + p
    p_mu = sum/N

    # calc covariance matrix
    def cov(x, y):
        xbar, ybar = x.mean(), y.mean()
        return np.sum((x - xbar) * (y - ybar)) / (np.size(x) - 1)

    def cov_mat(X):
        return np.array([[cov(X[0], X[0]), cov(X[0], X[1])], [cov(X[1], X[0]), cov(X[1], X[1])]])

    cov_matrix = np.sqrt(cov_mat(p_ls.T))

    y_min = np.min(p_ls.T[1])
    x_min = np.min(p_ls.T[0])
    y_max = np.max(p_ls.T[1])
    x_max = np.max(p_ls.T[0])

    #plot the results
    for p in p_ls:
        plt.plot(p[0], p[1], 'b+', markersize = 1, alpha = 1)
    plot_gauss_contour(p_mu, cov_matrix, x_min-pad, x_max+pad, y_min-pad, y_max+pad, title)

    plt.plot(p_true[0, 0], p_true[0, 1], 'gx')
    plt.text(p_true[0, 0] + 0.02, p_true[0, 1] + 0.02, r'$p_{true}$')

    # TODO calculate error measures and create plots----------------
    pass
#--------------------------------------------------------------------------------
def position_estimation_numerical_ml(data,nr_anchors,p_anchor, lambdas, p_true):
    """ estimate the position by using a numerical maximum likelihood estimator
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2 
        lambdas... estimated parameters (scenario 3), nr_anchors x 1
        p_true... true position (needed to calculate error), 2x2 """
    #TODO
    pass
#--------------------------------------------------------------------------------
def position_estimation_bayes(data,nr_anchors,p_anchor,prior_mean,prior_cov,lambdas, p_true):
    """ estimate the position by accounting for prior knowledge that is specified by a bivariate Gaussian
    Input:
         data...distance measurements to unkown agent, nr_measurements x nr_anchors
         nr_anchors... scalar
         p_anchor... position of anchors, nr_anchors x 2
         prior_mean... mean of the prior-distribution, 2x1
         prior_cov... covariance of the prior-dist, 2x2
         lambdas... estimated parameters (scenario 3), nr_anchors x 1
         p_true... true position (needed to calculate error), 2x2 """
    # TODO
    pass
#--------------------------------------------------------------------------------
def least_squares_GN(p_anchor,p_start, measurements_n, max_iter, tol, axs):
    """ apply Gauss Newton to find the least squares solution
    Input:
        p_anchor... position of anchors, nr_anchors x 2
        p_start... initial position, 2x1
        measurements_n... distance_estimate, nr_anchors x 1
        max_iter... maximum number of iterations, scalar
        tol... tolerance value to terminate, scalar"""

    rows = np.size(measurements_n)
    cols = np.size(p_start)

    p = p_start # original guess for B
    p = p.T
    Jf = np.zeros((rows, cols))  # Jacobian matrix from r
    r = np.zeros((rows, 1))  # r equations

    def distance(p0,p1):
        assert(p0.shape == (2,))
        assert(p1.shape == (2,))
        return np.sqrt(np.power((p0[0]-p1[0]),2)+np.power((p0[1]-p1[1]),2))

    def partialDerX(x,a,comDen):
        return (a-x)/comDen;

    def partialDerY(y,b,comDen):
        return (b-y)/comDen;
    points = []
    for iteration in range(max_iter):
        for anchor in range(rows):
            commonDenom = distance(p, p_anchor[anchor])
            assert(commonDenom != 0)
            Jf[anchor, 0] = partialDerX(p[0], p_anchor[anchor,0], commonDenom)
            Jf[anchor, 1] = partialDerY(p[1], p_anchor[anchor,1], commonDenom)
            r[anchor, 0] = measurements_n[anchor] - distance(p, p_anchor[anchor])

        Jft = Jf.T
        p_old = p
        p = p - np.dot(np.dot(inv(np.dot(Jft, Jf)), Jft), r).reshape(2,)
        assert (np.shape(p) == (2,))
        points.append(p)

        if distance(p_old, p)<tol:
            break;
        pass

    # plot converging
    if plot_converging:
        line = []
        for point in points:
            line.append(distance(p, point))
        random_color = np.random.rand(3,)
        axs[0].plot(line, c=random_color,linewidth=1, markersize=2, alpha=0.4)
        axs[1].plot(line, c=random_color,linewidth=1, markersize=2, alpha=0.4)

    return p

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Helper Functions
#--------------------------------------------------------------------------------
def plot_gauss_contour(mu,cov,xmin,xmax,ymin,ymax,title="Title"):
    
    """ creates a contour plot for a bivariate gaussian distribution with specified parameters
    
    Input:
      mu... mean vector, 2x1
      cov...covariance matrix, 2x2
      xmin,xmax... minimum and maximum value for width of plot-area, scalar
      ymin,ymax....minimum and maximum value for height of plot-area, scalar
      title... title of the plot (optional), string"""
	#npts = 100
    delta = 0.025
    X, Y = np.mgrid[xmin:xmax:delta, ymin:ymax:delta]
    pos = np.dstack((X, Y))
                    
    Z = stats.multivariate_normal(mu, cov)
    plt.plot([mu[0]],[mu[1]],'r+') # plot the mean as a single point
    plt.gca().set_aspect("equal")
    CS = plt.contour(X, Y, Z.pdf(pos),3,colors='r', alpha = 0.8,zorder=100)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)
    #plt.show()
    return

#--------------------------------------------------------------------------------
def ecdf(realizations):   
    """ computes the empirical cumulative distribution function for a given set of realizations.
    The output can be plotted by plt.plot(x,Fx)
    
    Input:
      realizations... vector with realizations, Nx1
    Output:
      x... x-axis, Nx1
      Fx...cumulative distribution for x, Nx1"""
    x = np.sort(realizations)
    Fx = np.linspace(0,1,len(realizations))
    return Fx,x

#--------------------------------------------------------------------------------
def load_data(scenario):
    """ loads the provided data for the specified scenario
    Input:
        scenario... scalar
    Output:
        data... contains the actual measurements, nr_measurements x nr_anchors
        reference.... contains the reference measurements, nr_measurements x nr_anchors"""
    data_file = 'measurements_' + str(scenario) + '.data'
    ref_file =  'reference_' + str(scenario) + '.data'
    
    data = np.loadtxt(data_file,skiprows = 0)
    reference = np.loadtxt(ref_file,skiprows = 0)
    
    return (data,reference)
#--------------------------------------------------------------------------------
def plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref=None, use_exp=True):
    """ plots all anchors and agents
    Input:
        nr_anchors...scalar
        p_anchor...positions of anchors, nr_anchors x 2
        p_true... true position of the agent, 2x1
        p_ref(optional)... position for reference_measurements, 2x1"""
    # plot anchors and true position
    plt.axis([-6, 6, -6, 6])
    for i in range(0, nr_anchors):
        if use_exp or (not use_exp and i!=0):
            plt.plot(p_anchor[i, 0], p_anchor[i, 1], 'bo')
            plt.text(p_anchor[i, 0] + 0.2, p_anchor[i, 1] + 0.2, r'$p_{a,' + str(i) + '}$')
    #plt.plot(p_true[0, 0], p_true[0, 1], 'g*')
    #plt.text(p_true[0, 0] + 0.2, p_true[0, 1] + 0.2, r'$p_{true}$')
    if p_ref is not None:
        plt.plot(p_ref[0, 0], p_ref[0, 1], 'r*')
        plt.text(p_ref[0, 0] + 0.2, p_ref[0, 1] + 0.2, '$p_{ref}$')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    #plt.show()
    pass

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
