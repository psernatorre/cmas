function [F] = single_agent_output_feedback(A, B, C, D, Q, R, P)

n = length(A);
m = width(B);
ny = height(C);

yalmip('clear')

F = sdpvar(m,ny, 'full');

alpha = sdpvar(1,1, 'full');

N = P * B * (F * D * inv(R) + inv(R) * D' * F') * B' * P;
B_bar = B * (eye(m) + F * D);

constraints = [[Q - P * B * F * C - C' * F' * B' * P + N,     P * B_bar;
                B_bar' * P,                                     R ] >= alpha * eye(n+m)];

% Minimize the parameter alpha. In theory, it should be zero.
objective_function = -alpha;

% Set some options for YALMIP and solver
options = sdpsettings ( 'verbose' ,0, 'solver', 'mosek') ;

% Solve the problem
opt_model = optimize ( constraints , objective_function , options ) ;
    
sdp_status = opt_model.problem;

F = value(F);
alpha = value(F);

return