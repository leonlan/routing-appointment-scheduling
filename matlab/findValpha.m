function [V, alpha, param] = findValpha(mean_i, scv)
%N=5      
    if scv < 1
    k = 1;
    while (scv < 1/k)
        k = k+1;
    end
    p = (k * scv - sqrt(k * (1 + scv) - k^2 * scv))/(1 + scv);
    mu = (k - p)/mean_i;
    V = -mu * eye(k);
    for j=1:k-2
        V(j,j+1) = mu;
    end
    V(k-1,k) = (1-p)*mu ;
    alpha = zeros(1, k);
    alpha(1) = 1-p;
    alpha(2) = p;
    param = [mu, p, k];
 else
    V = zeros(2);
    alpha = zeros(1,2);
    alpha(1) = (1 + sqrt((scv - 1)/(scv + 1)))/2;
    alpha(2) = 1 - alpha(1);
    V(1,1) = -2*alpha(1)/mean_i; %-mu1
    V(2,2) = -2*alpha(2)/mean_i; %-mu2  
    param = [-V(1,1), -V(2,2), alpha(1)];
end
end 