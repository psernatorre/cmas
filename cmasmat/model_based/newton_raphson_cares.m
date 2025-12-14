function [Pfin, F, P, CARE, Acl] = newton_raphson_cares(A,B,Q,R,niter, solver)

% Function that solves coupled algebraic Riccati equations (CAREs) using
% newton-raphson method and optimization problem.

% Description of the method and references to cite:
% The algorithm is based on solving linear matrix inequalities that can 
% derived by the CAREs. This approach is presented by:
% Jacob Engwerda, 'Algorithms for computing Nash equilibria in deterministic
% LQ games', 2006.
% We have built on the previous algorithm. We have reformulated the 
% linear matrix equalities as an optimization problem. Moreover, 
% some additional instructions have been added.
% This work has been published as part of the toolbox presented by:
% Paul Serna-Torre, and Patricia Hidalgo-Gonzalez, 'Static Output Feedback 
% Control for Multi-Agent Systems Using Nash Equilibrium', 2025.

% Description of the input data:
% A: state matrix of the multi-agent system.
% B: cell that contains agents' input matrices, i.e., B{1}, ..., B{N}, 
% where N is the number of agents. It should be created 
% as B = cell(number of agents, 1).
% Q: cell that contains agents' state cost matrix, i.e., Q{1}, ..., Q{N}.
% It should be created as B.
% R: cell that contains agents' input cost matrix, i.e., R{1}, ..., R{N}. 
% It should be created as B.
% niter: number of iterations. You should test out with a certain number,
% and then see if the Max(|CARE|) is very low, like 2.53e-10. 
% solver: solver installed in your MATLAB configuration. For instance, you
% can use 'mosek'.

% Description of printout in command window when the function is executed:
% Iteration: number of iteration.
% SDP status: status of the optimization model after the solver 
% has been executed. For instance, '0' means successfully solved. Please,
% refer to YALMIP error status.
% SDP alpha: value of the decision variable alpha. Ideally, this alpha
% should be very close to zero because the linear matrix equalities should
% be zero.
% Max(|CARE|): maximum of the entries of the entry-wise absolute value of the
% left-hand side of the CAREs. This value should decrease to zero as the iteration
% goes.
% Re(Lambda): maximum of the real parts of the eigenvalues of the
% closed-loop state matrix. This value should be negative, thus indicating
% that the system is stable.

% Description of the output data:
% Pfin: cell that contains the set of matrices that solves the CAREs in the
% final iteration. It is of the form P{1}, ..., P{N}, where N is the number
% of agents.
% F: state-feedback controller using the solution of the Riccati equation. 
% It is of the form F{1}, ..., F{N}.
% CARE: left-hand side of the CARE in the final iteration.
% Acl: state matrix of the closed-loop system using the state-feedback
% control F.

nag = length(B); 
nst = length(A);
P = cell(nag,niter);
CARE = cell(nag,1);
maxCARE = zeros(nag,1);
S = cell(nag,1);
F = cell(nag,1);
Pfin = cell(nag,1);


fprintf('====================================\n');
fprintf('Newton-Raphson method to solve CAREs \n');
fprintf('====================================\n');

for j=1:nag
    S{j} = B{j}*(R{j}^(-1))*B{j}';
end

fprintf('%9s%2s%10s%2s%9s%2s%11s%2s%10s\n', 'Iteration', blanks(2),'SDP status',blanks(2), "SDP alpha", blanks(2),'Max(|CARE|)',blanks(2),'Re(Lambda)');
for i=1:niter

    if i==1

        % Initialization of the algorithm using LQR
        [~, P_lqr, ~] = lqr(A,horzcat(B{:}),1/nag*sum(cat(3,Q{:}),3), blkdiag(R{:}));

        for j=1:nag
            P{j,i} = P_lqr;
        end       
        sdp_status = 0;
        alpha = 0;

    else
    
        % Define stabilizing solutions to CAREs
        for j=1:nag
            P{j,i} = sdpvar(nst,nst, 'symmetric');
        end

        % Define slack parameter
        alpha = sdpvar(1,1, 'symmetric');

        % Define constraints
        constraints = [];

        for j=1:nag
            sumx = 0;
            for k =1:nag
                if k~=j
                    sumx = sumx + -P{k,i}*S{k}*P{j,i-1} + P{k,i-1}*S{k}*P{j,i-1} + (-P{k,i}*S{k}*P{j,i-1} + P{k,i-1}*S{k}*P{j,i-1})';
                end
            end

            LHS = Acl'*P{j,i} + P{j,i}*Acl + sumx + P{j,i-1}*S{j}*P{j,i-1} + Q{j};

            % The idea is to make LHS equal to zero because it is a CARE.
            % However, using ==0 could be very strict. Instead, we use the
            % parameter 'alpha'.

            for c=1:nst
                for r = c:nst
                    constraints = [constraints, abs(LHS(r,c)) <= alpha];
                end
            end
            constraints = [constraints, P{j,i}>=0];
            constraints = [constraints, alpha>=0];
        end
        
        % Minimize the parameter alpha. In theory, it should be zero.
        objective_function = alpha;

        % Set some options for YALMIP and solver
        options = sdpsettings ( 'verbose' ,0, 'solver', solver ) ;

        % Solve the problem
        opt_model = optimize ( constraints , objective_function , options ) ;
    
        sdp_status = opt_model.problem;

        % Retrieve optimal solution
        for j=1:nag
            P{j,i} = value(P{j,i});
            alpha = value(alpha);
        end

    end

    % State-matrix of the closed-loop system. This is then used in the next
    % iteration
    sumx = 0;
    for j=1:nag
        sumx = sumx - S{j}*P{j,i};
    end
    Acl = A + sumx; 

    % Maximum of the eigenvalue real parts. This is to check stability.
    maxReEig = max(real(eig(Acl)));
    
    % Calculate the left-hand side of the CARE for each agent. In theory,
    % every CARE should be equal to zero.

    for k = 1:nag
        Pfin{k} = P{k,i};
        CARE{k} = Acl'*Pfin{k} + Pfin{k}*Acl + Pfin{k}*S{k}*Pfin{k} + Q{k}; 
        maxCARE(k) = max(max(abs(CARE{k})));
        F{k} = -R{k}^(-1)*B{k}'*Pfin{k};
    end
    fprintf("%7s%2d%11s%1d%3s%2.2e%5s%3.2e%3s%2.2e\n", blanks(7),i, blanks(11),sdp_status, blanks(2), alpha, blanks(5), max(maxCARE), blanks(3), maxReEig); 
end

fprintf('\n')
fprintf('Closed-loop system eigenvalues: \n')
display(eig(Acl))
fprintf('\n')

end

