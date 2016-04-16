import numpy as np

def bspline(x,lbound,rbound,knots=20,degree=3):
    """B-Spline basis from Eilers-Marx
    IN:
        x       -   1-D array for which the basis should be computed
        lbound  -   left bound for B-spline basis, if None then min(x) is used
        rbound  -   right bound for B-spline basis, if None then max(x) is used
        knots   -   number of knots to use
        degree  -   degree of B-spline basis
    OUT:
        B       -   n X b array, n = len(x), b = knots + degree - 1"""
    dx = (rbound-lbound)/knots
    t = lbound + dx * np.arange(-degree,knots-1)
    T = np.outer(np.ones(x.shape),t)
    X = np.outer(x,np.ones(t.shape))
    P = (X-T)/dx
    B = (T<=X) & (X<(T+dx))
    r = np.append(np.arange(2,t.shape[0]+1),1)
    for k in xrange(1,degree+1):
        B = (P*B + (k+1-P)*B[:,r-1])/r
    return B

def smoothRF( X,Y,nrows,ncols ,nknots_x,nknots_y,pord_x,pord_y,lambda_x,lambda_y):
    """smoothRF Estimate a receptive field using a smoothness constraint
    %   IN:
    %       X           -       N x M matrix of lagged spectrograms, N rows/timepoints and 
    %                           M nrows*ncols columns/features. Each column is
    %                           assumd to have mean=0 and sd=1
    %       Y           -       N x 1 vector to be predicted. Is assumed to have
    %                           zero mean (therefore no bias term is needed in X
    %                           and RF)
    %       nrows       -       Number of rows of receptive field, equal to the
    %                           number of frequency features/y dimension of RF
    %       ncols       -       Number of colums of receptive field, equal to the
    %                           number of lags/x dimension of RF
    %       nknots_x    -       Number of knots of x dimension in spline basis. 
    %                           Ideally close to ncols - 4 (degree of cubic spline)
    %       nknots_y    -       Number of knots of y dimension in spline basis.
    %                           Ideally close to nrows - 4 (degree of cubic spline)
    %       pord_x      -       Order of differences for penalty matrix of x dimension of RF.
    %                           2 is a good value to penalize roughness, but other values are sometimes used.
    %       pord_y      -       Order of differences for penalty matrix of y dimension of RF.
    %                           2 is again a good default value.
    %       lambda_x    -       Smoothness penalty in x dimension of RF
    %       lambda_y    -       Smoothness penalty in y dimension of RF
    %
    %   OUT:
    %       RF          -       Estimated (smooth) receptive field (flattened)"""
    #use a cubic spline
    bdeg = 3
    #Create B-spline basis of RF, a small constant (0.1) is added and
    #substracted to prevent problems at the edges of the spline basis
    B1 = bspline(np.arange(1,nrows+1),0.9,nrows+0.1,nknots_y,bdeg)
    B2 = bspline(np.arange(1,ncols+1),0.9,ncols+0.1,nknots_x,bdeg)
    m1,n1 = B1.shape
    m2,n2 = B2.shape
    #create difference matrices
    D1 = np.diff(np.eye(n1), pord_y,0)
    D2 = np.diff(np.eye(n2), pord_x,0)
    #create penalty matrices for each dimension
    P1 = np.kron(np.eye(n1),D1.T.dot(D1))
    P2 = np.kron(D2.T.dot(D2),np.eye(n2))
    
    #weight penalty matrices by dimension-specific smoothing factor
    P = lambda_y*P1 + lambda_x*P2
    
    #create 2-dimensional kronecker product B-spline of B1 and B2
    B =np.kron(B2,B1)
    
    #create new 'data' matrix U
    U = X.dot(B)
    
    #estimate B-spline coefficients through normal equations
    #could in principle be replaced by gradient descent or cholesky
    #decomposition, but seems to be reasonably fast already
    gammas = np.linalg.solve((U.T.dot(U)+P),U.T.dot(Y))
    
    #now create receptive field
    return B.dot(gammas)
    
    
    
