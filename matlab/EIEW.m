function [val] = EIEW(x, alpha, Vn, invVn, N, omega_b, p)
% interarrival times are given by the vector x and the sojourn time
N = size(x,1);
%m = size(alpha,2);
obj = zeros(N,1);
PalphaF = alpha;
m = size(alpha,2);

%x(0)=0;
for i=1:N
  % obj(i) = x(i)+sum(PalphaF*Vn(1:i*m,1:i*m),2);
   % obj(i) = x(i)+sum(PalphaF*invVn(1:i*m,1:i*m),2);
    obj(i) = -(sum(PalphaF*invVn(1:i*m,1:i*m)*expm(Vn(1:i*m,1:i*m)*x(i)),2))+omega_b*(x(i)+sum(PalphaF*invVn(1:i*m,1:i*m),2));
    P = PalphaF* expm(Vn(1:i*m, 1:i*m) * x(i));
    F = 1 - sum(P);
    PalphaF = horzcat(P, alpha*F) + (1-p) * horzcat(zeros(1, m), P);
end

%ES_N = -PalphaF .* sum(invVn,2);
val = sum(obj);
%makespan = sum(x) + ES_N;
end