function Vnew = createVn(i, V, Vne, Vold, alpha, N)

if i==1
    Vnew=V;
%    n = size(V,1);
 %   Vnew = zeros((N+1) * n);
  %  V_b = -sum(V,2) * alpha;
   % Vnew(1:n, 1:n) = V;
    %for j=1:N;
     % Vnew(j*n+1: (j+1)*n, j*n + 1:(j+1)*n) = V;
      %Vnew((j-1)*n + 1: j*n, j*n+1: (j+1)*n) = V_b;
      alpha;
      Vnew;
else
%n = size(V,1);
 %   Vnew = zeros((N+1) * n);
  %  V_b = -sum(Vold,2) * alpha;
   % Vnew(1:n, 1:n) = V;
   %for j=1:N;
    %  Vnew(j*n+1: (j+1)*n, j*n + 1:(j+1)*n) = V;
     % Vnew((j-1)*n + 1: j*n, j*n+1: (j+1)*n) = V_b;
   %end
n = size(V,1);
n2=size(Vne, 1);
Vnew = zeros(n+n2);
Vold;
alpha;
V_b = -sum(Vold,2) * alpha;
n3=size(V_b,1);
    Vnew(1:n2, 1:n2) = Vne;
    Vnew(n2+1:n+n2, n2+1:n+n2)=V;
    b=n2-n3+1;
    Vnew(b :n2,n2+1:n+n2)=V_b;
    %Vnew(n2-n3)
    V_b;
    Vnew;
   %for j=1:N;
    %  Vnew(j*n+1: (j+1)*n, j*n + 1:(j+1)*n) = V;
     % Vnew((j-1)*n + 1: j*n, j*n+1: (j+1)*n) = V_b;
   %end
   
    %n = size(V,1);
    %n2=size(Vne, 1);
    %Vnew = zeros(n+n2);
    %Vnew = zeros((N+1) * n);
    %V_b = -sum(Vold,2) * alpha;
    %Vnew(1:n2, 1:n2) = Vne;
    %Vnew(n2+1:n+n2, n2+1:n+_n2)=V;
   %for j=1:N;
    %  Vnew(j*n+1: (j+1)*n, j*n + 1:(j+1)*n) = V;
     % Vnew((j-1)*n + 1: j*n, j*n+1: (j+1)*n) = V_b;
   %end
    %V1=Vnew;
end
%Va=V;
end 




