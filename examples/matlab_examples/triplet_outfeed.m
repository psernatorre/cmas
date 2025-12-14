clear
clc
close all
yalmip('clear')

%{
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
%}

A = [-0.0366, 0.0271, 0.0188, -0.4555;
     0.0482, -1.0100, 0.0024, -4.0208;
     0.1002, 0.3681, -0.7070, 1.4200;
     0,          0,      1,      0];

B = [0.4422, 0.1761;
     3.5446, -7.5922;
     -5.52, 4.49;
     0, 0];

C = [0, 1, 0, 0];

D = zeros(height(C), width(B));

Q = eye(4);
R = eye(2);

% Solution Pi for CAREs
[~, P, ~] = lqr(A, B, Q, R);


F = single_agent_output_feedback(A, B, C, D, Q, R, P);

alpha_coef = 1;
beta_coef = 0;
gamma_coef = 0;

[F, alpha, beta, gamma, opt_model, Acl_F] = mas_output_feedback(A, {B}, {C}, {D}, {Q}, {R}, {P}, alpha_coef, beta_coef, gamma_coef, 'mosek');

F{1}