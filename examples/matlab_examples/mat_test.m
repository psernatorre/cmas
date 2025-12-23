clc
clear
A = -1*eye(2);

nag = 2;

B = cell(nag, 1);
B{1} = [1; 0];
B{2} = [0; 1];

Q = cell(nag, 1);
Q{1} = 2*eye(2);
Q{2} = 1*eye(2);

R = cell(nag, 1);
R{1} = 1;
R{2} = 2;

niter = 3;

[P, F, CARE, Acl] = lyapunov_iteration_cares(A, B, Q, R, niter)