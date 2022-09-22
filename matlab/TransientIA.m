function [x, fval, feval] = TransientIA(mean_i, scv, omega_b, N, n, x0, p, ~)

if nargin > 6; tic; end

for i=1:N
    %mean_i(i)
    %scv(i)
    [V, alpha, ~] = findValpha(mean_i(i), scv(i))
    alpha;
    Va=V;
    if i==1
        Vne=V;
        Vold=V;
    end
    Vn = createVn(i, V, Vne, Vold, alpha, N)
    Vne=Vn;
    Vold=V;
    invVn = inv(Vn);
end
Vn
alpha;
%Setting the correct options and starting value
options = optimset('Display', 'off', ...
                   'TolX', 1e-15);
x0=[];
if isempty(x0); 
    x0 = 1.5*ones(i,1); 
end
%x0 = 1.5*ones(i,1); 
[x, fval, ~, output] = fmincon(@(x) EIEW(x, alpha, Vn, invVn, N, omega_b, p), ...
    x0, -eye(i), zeros(i,1), [], [], [], [], [], options);
%x
%[fval2] = Transient_EIEW(x, alpha, Vn, invVn, omega_b, p);
%fs(i)=fval;
%val = sum(obj);
if nargin > 6; toc;
    disp(['The optimization took ', num2str(output.('funcCount')),...
    ' function evaluations']);
    
end
feval = output.('funcCount');

%fval=sum(fs);
end