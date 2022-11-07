function delayembed(X, numdelays::Int)
    """
        arguments: 
        X: n × m data array with n spatial points and m timepoints
        numdelays: integer, number of delays

        returns:
        Xemb: ((numdelays + 1)⋅n) × (m - numdelays) data array 
        (a delay embedded array)

        appears to work for the following:
        testMat = [1 2 3 4 5 6 ; 7 8 9 10 11 12 ; 13 14 15 16 17 18];
        k = 0, 1, 2, 3
        delayembed(testMat,k)
        delayembed(testMat,0) == testMat
    """
        n = size(X,1);
        m = size(X,2);
        Xemb = zeros(typeof(X[1]), n*(1 + numdelays), m - numdelays);
        for del in 0:numdelays
            Xemb[del*n .+ (1:n), :] = X[:, (1 + del):(m - numdelays + del)];
        end
        
        return Xemb
end