%omega = 0;
omega_b=0.8;
N = 10;
n=1;

mean_i=0.5*ones(1,N);

scv=0.5*ones(1,N);
%scv(21:30)=1.5; %IA3
%scv(1:5)=1.2;
%[ 0.2 0.3 0.5 1.1 1.2]
%TransientIA(scv, omega_b, N, n, [], 1, 1);
%for i=1:N
[IA, fval]=TransientIA_htp(mean_i, scv, omega_b, N, n, [], 1, 1)


IA;
fval;








