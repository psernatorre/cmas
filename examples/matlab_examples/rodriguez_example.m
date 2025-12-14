% Here, the proposed control is compared with the framework in Rodrigues, 
% From lqr to static output feedback: a new lmi approach, 2022.  
% Example 4.1 of the mentioned paper is reproduced. 

clear
clc
close all
yalmip('clear')

% Definition of state-space matrices and cost matrices
A = [0 1 0;
 0 1 1;
     0 13 0];
B{1} = [0;0;1]; % there is only one agent
C{1} = [0  5 -1;
        -1 -1 0];
D{1} = [0;
        0];

Q{1} = blkdiag(1,3,0.1);
R{1} = 10^(-4);

% Solution Pi for CAREs
niter = 2;
[Pn, Fn, ~, ~, Acl_n] = newton_raphson_cares(A,B,Q,R,niter, 'mosek');

% Synthesis of output feedback control using proposed framework
% The alpha, beta, and gamma coefficients have been set to have the
% solution that Rodrigues provides in the paper.

alpha_coef = 1;
beta_coef = 0;
gamma_coef = 0;

[F, alpha, beta, gamma, opt_model, Acl_F] = mas_output_feedback(A, B, C, D, Q, R, Pn, alpha_coef, beta_coef, gamma_coef, 'mosek');


% Definition of time and initial conditions for simulation
time_steps= 0:0.01:20;
IC = [1; 1; 1];

% Closed-loop system with the output feedback control F
sys_F = ss(Acl_F, zeros(3,1), eye(3), zeros(3,1));

% Simulation of the closed-loop system 
[y_F,t] = initial(sys_F,IC,time_steps);

% Calculation of the objective function
f = cell(1,1);
for j=1:1
    u = F{j}*C{j}*y_F';
    f{j} = cost_calculation(y_F, u', t, Q{j}, R{j});
end

% Output feedback control given in Rodrigues' paper
F_rd{1} = [6.8981 84.9224];

% Closed-loop system with Rodrigues' control
Acl_rd = A + B{1}*F_rd{1}*C{1};
sys_F_rd = ss(Acl_rd, zeros(3,1), eye(3), zeros(3,1));

% Simulation of the closed-loop system using Rodrigues' control
[y_Frd,t] = initial(sys_F_rd,IC,time_steps);

% Calculation of the objective function using Rodrigues' control
f_rd = cell(1,1);
for j=1:1
    u = F_rd{j}*C{j}*y_Frd';
    f_rd{j} = cost_calculation(y_Frd, u', t, Q{j}, R{j});
end

% Note: the idea is to compare F_rd with F, f with f_rd. You will see that
% they are almost the same.