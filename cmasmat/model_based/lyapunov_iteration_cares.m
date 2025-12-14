function [P, F, CARE, Acl] = lyapunov_iteration_cares(A,B,Q,R,niter)

nag = length(B);
S = cell(nag,1);

fprintf('====================================\n');
fprintf('Lyapunov iterations to solve CAREs \n');
fprintf('====================================\n');

fprintf('%9s%2s%11s%2s%10s\n', 'Iteration', blanks(2), 'Max(|CARE|)',blanks(2),'Re(Lambda)');

Acl_m = cell(nag,1);
P = cell(nag,1);
F = cell(nag,1);
CARE = cell(nag,1);
maxCARE = zeros(nag,1);

for j=1:nag
    S{j} = B{j}*(R{j}^(-1))*B{j}';
    Acl_m{j} = A;
end

for i=1:niter

    for j=1:nag
        [P{j},~,~,info] = icare(Acl_m{j},[],Q{j},[],[],[],-S{j});
    end

    for j=1:nag
        sumx = 0;
        for k=1:nag
                if k~=j
                    sumx = sumx  - S{k}*P{k};
                end
        end
        Acl_m{j} = A + sumx;
        Acl = A + sumx - S{j}*P{j};
        maxReEig = max(real(eig(Acl)));
        CARE{j} = Acl_m{j}'*P{j} + P{j}*Acl_m{j} - P{j}*S{j}*P{j}+ Q{j} ;
        maxCARE(j) = max(max(abs(CARE{j})));
        F{j} = -R{j}^(-1)*B{j}'*P{j};
    end

   fprintf("%7s%2d%5s%3.2e%3s%2.2e\n", blanks(7),i, blanks(5), max(maxCARE), blanks(3), maxReEig); 
end

fprintf('Closed-loop system eigenvalues: \n')
display(eig(Acl))
fprintf('\n')

end

