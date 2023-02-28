%    Coursework 2023 -- Part 2
clc;
clear;
%% Main Body -- Do NOT edit

if ~exist('suppress_figures','var')
    clc; clear; close all;
    show_plots =    1;
else
    show_plots =    0;
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%           Section 1          %%%%%%%%%%%%%%
%%%%%%%%%%%%%%          PERCEPTRON          %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('dataset_heart_attack'); % loads X, t and x_titles
I =                 400; % trainingnumber of  iterations
gamma =             0.2; % learning rate
theta_A =           [1.83388501459509;-2.25884686100365;0.862173320368121;0.318765239858981;-1.30768829630527;-0.433592022305684;0.342624466538650;3.57839693972576;2.76943702988488;-1.34988694015652;3.03492346633185;0.725404224946106;-0.0630548731896562;0.714742903826096];
theta_B =           [-0.204966058299775;-0.124144348216312;1.48969760778547;1.40903448980048;1.41719241342961;0.671497133608081;-1.20748692268504;0.717238651328839;1.63023528916473;0.488893770311789;1.03469300991786;0.726885133383238;-0.303440924786016;0.293871467096658;-0.787282803758638;0.888395631757642;-1.14707010696915;-1.06887045816803;-0.809498694424876;-2.94428416199490;1.43838029281510;0.325190539456198;-0.754928319169703;1.37029854009523;-1.71151641885370;-0.102242446085491;-0.241447041607358];

% Function 1

t_hat_A=            perceptron_predict(     X,      @mapping_A,  theta_A );
t_hat_B=            perceptron_predict(     X,      @mapping_B,  theta_B );
% Function 2
cm_A =              classification_margin(  X, t,   @mapping_A,  theta_A );
cm_B =              classification_margin(  X, t,   @mapping_B,  theta_B );
% Function 3
grad_per_theta_A =  perceptron_gradient(    X, t,   @mapping_A,  theta_A );
grad_per_theta_B =  perceptron_gradient(    X, t,   @mapping_B,  theta_B );
% Function 4
loss_haz_A =        hinge_at_zero_loss(     X, t,   @mapping_A,  theta_A );
loss_haz_B =        hinge_at_zero_loss(     X, t,   @mapping_B,  theta_B );
% Function 5
theta_mat_A =       perceptron_train_sgd(   X, t,   @mapping_A,  theta_A, I, gamma );
theta_mat_B =       perceptron_train_sgd(   X, t,   @mapping_B,  theta_B, I, gamma );

if show_plots % perceprton
    vec_loss_haz_A =            zeros(I+1,1);
    vec_loss_haz_B =            zeros(I+1,1);
    for ii=1:(I+1)
        vec_loss_haz_A(ii) =    hinge_at_zero_loss(  X,   t, @mapping_A, theta_mat_A(:,ii) );
        vec_loss_haz_B(ii) =    hinge_at_zero_loss(  X,   t, @mapping_B, theta_mat_B(:,ii) );
    end

    figure;
    plot(0:I,vec_loss_haz_A,'mo-' ,'DisplayName','features A'); hold on;
    plot(0:I,vec_loss_haz_B,'gx-' ,'DisplayName','features B');
    xlabel('Iteration','interpreter','latex');
    ylabel('loss','interpreter','latex');
    title ('Perceptron hinge-at-zero loss','interpreter','latex');
    legend show; grid;
end

% Function 6
discussion();

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%           Section 2          %%%%%%%%%%%%%%
%%%%%%%%%%%%%%     LOGISTIC REGRESSION      %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S =                     30; % mini-batch size

% Function 7
logit_hat_A=            logistic_regression_logit(      X,      @mapping_A,  theta_A );
logit_hat_B=            logistic_regression_logit(      X,      @mapping_B,  theta_B );
% Function 8 
grad_lr_theta_A =       logistic_regression_gradient(   X, t,   @mapping_A,  theta_A );
grad_lr_theta_B =       logistic_regression_gradient(   X, t,   @mapping_B,  theta_B );
% Function 9
loss_lr_A =             logistic_loss(                  X, t,   @mapping_A,  theta_A );
loss_lr_B =             logistic_loss(                  X, t,   @mapping_B,  theta_B );
% Function 10
theta_lr_mat_A =        logistic_regression_train_sgd(  X, t,   @mapping_A,  theta_A, I, gamma, S );
theta_lr_mat_B =        logistic_regression_train_sgd(  X, t,   @mapping_B,  theta_B, I, gamma, S );

% code for evalutating the test loss while training
vec_loss_lr_A =         zeros(1,I+1);
vec_loss_lr_B =         zeros(1,I+1);
for ii=1:(I+1)
    vec_loss_lr_A(ii) = logistic_loss(                  X, t,   @mapping_A,  theta_lr_mat_A(:,ii) );
    vec_loss_lr_B(ii) = logistic_loss(                  X, t,   @mapping_B,  theta_lr_mat_B(:,ii) );
end

% auxiliary code for plotting
if show_plots % logistic regression
    figure;
    plot(0:I,vec_loss_lr_A,'mo-','DisplayName','features A'); hold on;
    plot(0:I,vec_loss_lr_B,'gx-','DisplayName','features B');
    xlabel('Iteration','interpreter','latex');
    ylabel('loss','interpreter','latex');
    title ('Logistic Regression loss','interpreter','latex');
    legend show; grid;
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%           Section 3          %%%%%%%%%%%%%%
%%%%%%%%%%%%%%      NEURAL NETWORKS         %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

theta_nn.W1 =               [-0.9,+0.2;-0.5,+1.7;+0.4,+0.8];
theta_nn.W2 =               [-0.6,-1.1,-0.9;+1.4,-0.6,-0.2];
theta_nn.w3 =               [-0.8;+1.8];
x1_nn =                     [-1.0;+2.5];
t1_nn =                     1;
x2_nn =                     [-0.4;+3.2];
t2_nn =                     0;
% Function 11
leaky_ReLU_nn_1 =           leaky_ReLU(             x1_nn);
leaky_ReLU_nn_2 =           leaky_ReLU(             x2_nn);
% Function 12
grad_leaky_1 =              grad_leaky_ReLU(        x1_nn);
grad_leaky_2 =              grad_leaky_ReLU(        x2_nn);
% Function 13
[logit_nn_1, grad_nn_1] =   nn_logit_and_gradient( x1_nn, t1_nn, theta_nn);
[logit_nn_2, grad_nn_2] =   nn_logit_and_gradient( x2_nn, t2_nn, theta_nn);


disp('Licenses used (make sure only ''matlab'' follows):');
license('inuse')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%   AUXILIARY FUNCTIONS        %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function U = mapping_A(X)
    vec_bias =  [131.62,246.26,0.15,0.53,149.65,0.33,1.04,1.4,0.73,2.31,54.37,0.68,0.97];
    vec_slope = [17.54,51.83,0.36,0.53,22.91,0.47,1.160,0.62,1.020,0.61,9.080,0.47,1.030];
    U =         [ones(size(X,1),1), (X-vec_bias)./vec_slope];
end
 
function U = mapping_B(X)
    vec_bias =  [131.62,246.26,0.15,0.53,149.65,0.33,1.04,1.4,0.73,2.31,54.37,0.68,0.97,17631.39,63323.53,0.15,0.55,22917.1,0.33,2.42,2.34,1.57,5.73,3037.91,0.68,2];
    vec_slope = [17.54,51.83,0.36,0.53,22.91,0.47,1.16,0.62,1.020,0.61,9.080,0.47,1.030,4858.22,29216.36,0.36,0.64,6609.27,0.47,4.520,1.580,3.060,2.710,975.830,0.47,2.640];
    U =         [ones(size(X,1),1), ([X,X.^2]-vec_bias)./vec_slope];
end

function out = sigmoid(in)
    out =       1./(1+exp(-in));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   INSERT YOUR CODE IN THE FUNCTIONS BELOW  %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function 1
function t_hat= perceptron_predict(X, map_func, theta)
%     z = X' * map_func(X) * theta;
    z=map_func(X)*theta;
    for i = 1:length(z)
        if z(i,1) == 0
            t_hat(i,1)=0;
        end
    end
    t_hat = hardlim(z);



    % theta_t X u(x)>0 --> t_hat = 1 
    
end

% Function 2
function cm= classification_margin(X, t, map_func, theta)
    % compute 
    t_cm = 2*t-1;
    
    U = map_func(X); %303*14

    cm =t_cm.*(U*theta);

end

% Function 3
function grad_theta = perceptron_gradient(X, t, map_func, theta)
    grad_theta =            zeros(size(X,1),length(theta));
    t_cm = 2*t-1;%303x1
    U = map_func(X); % 303x14
    z = U*theta;

    N = size(X,1);
    for i = 1:N
        if t_cm(i) * z(i) > 0
            grad_theta(i,:) = 0;
        else
            grad_theta(i,:) = -t_cm(i,1)* U(i,:); % 1x14 = 303x1 * 1*14
        end
    end



end

% Function 4 
function loss = hinge_at_zero_loss(X, t, map_func, theta)

    t_cm = 2*t-1;
    N = size(X, 1);
    sum = 0;

    U = map_func(X);
    t_hat =  U*theta ;
    for i = 1:N    
            sum = sum + max(0,-t_cm(i)*t_hat(i));
    end
    loss = sum / N;

   
end

% Function 5
function theta_mat = perceptron_train_sgd(X, t, map_func, theta_init, I, gamma)
    theta_mat =             zeros(length(theta_init),I+1);
    theta_mat(:,1) = theta_init;
    theta_temp = theta_init;
    t_cm_tr = 2 * t -1;
    U_tr = map_func(X);
    for i=1:I
        testnum = 1 + mod(i-1,size(t,1));
        predict = U_tr * theta_temp;
        if predict(testnum) * t_cm_tr(testnum) < 0
            theta_temp = theta_temp + (gamma * t_cm_tr(testnum) * U_tr(testnum,:))';
        end
        theta_mat(:,i+1) = theta_temp;
    end
end

% Function 6
function discussion()
    disp('discussion:');
    disp(['<No, we can not tell the whether its fitted performance. ' ...
        'Because if we want to judge this, we need the training loss as well as validating loss. But we only have training loss, so we cant judge. >']);
end

% Function 7
function logit= logistic_regression_logit(X, map_func, theta)

    logit = map_func(X)*theta;
   
end

% Function 8
function grad_theta = logistic_regression_gradient(X, t, map_func, theta)
    grad_theta = zeros(size(X,1),length(theta)); %303x14
    
    U = map_func(X); %303 *14

    logit = U*theta;  % 303x1
    
  for i = 1:length(theta)
   grad_theta(:,i) = (sigmoid(logit)-t) .* U(:,i);  %303x1 * 303*14

  end  

end

% Function 9
function loss = logistic_loss(X, t, map_func, theta)
   
    U = map_func(X);%303*14
    t_cm = 2*t-1; %303x1
    logit = U*theta;    %303x1
    N = size(t,1);
    sum = 0;

    for i = 1:N
        sum = sum - log(sigmoid(t_cm(i,1)*logit(i,1)));
            
    end
    loss = sum/N;
end

% Function 10
function theta_mat = logistic_regression_train_sgd(X, t, map_func, theta_init, I, gamma, S)

    theta_mat = zeros(length(theta_init),I+1);
    theta_mat(:,1) = theta_init;%14x401
    
    theta_temp = theta_init; % 14x1
%     logit = zeros(size(X,1),1);
    N = size(t,1);

    for i = 1:I
        Si = 1+mod(S*(i-1)+(0:S-1)+1,N);
        sum = zeros(size(theta_init));
        for j = 1:S
            n = Si(j);
            lg = logistic_regression_gradient(X(n,:),t(n),map_func,theta_mat(:,i));
%             sum = sum + logistic_regression_gradient(X(n,1),t(n),map_func,theta_mat(:,i)); % check  gad 1x14
            sum = sum +lg';
        end
        change = sum * gamma / S; % 14x1
        theta_temp = theta_temp - change;
        theta_mat(:,i+1) = theta_temp;
   
    end  
end


% Function 11
function out = leaky_ReLU(in)
    out = max(0.1*in,in);
end

% Function 12
function out = grad_leaky_ReLU(in)
% dh/da = l +al    
    out = in;
    for i  = 1:length(in)
        if in(i,1) > 0
            out(i,1)=1;
        else
            out(i,1)=0.1;
        end
    end

end


% Function 13
function [logit,grad_theta] = nn_logit_and_gradient(x, t, theta)
   grad_theta.w3 =         theta.w3;
   grad_theta.W2 =         theta.W2;
   grad_theta.W1 =         theta.W1;
 
   h0 = x;
   a1 = theta.W1 * h0;
   h1 = leaky_ReLU(a1);
   
   a2 = theta.W2 * h1;
   h2 = leaky_ReLU(a2);

   a3 = theta.w3' * h2;

   logit = sigmoid(a3);%0.3826

% logi loss f (y) = log(1 + exp(−y)).
   delta.w3 = sigmoid(a3) - t;
   delta.W2 = (grad_leaky_ReLU(a2).*theta.w3) * delta.w3;
   delta.W1 = grad_leaky_ReLU(a1).*(theta.W2'* delta.W2);

   grad_theta.w3 = delta.w3 * h2;
   grad_theta.W2 = delta.W2 * h1';
   grad_theta.W1 = delta.W1 * h0';

   
end