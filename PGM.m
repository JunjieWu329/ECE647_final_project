clear; close all; clc;
rng(1234);  % Set RNG seed for reproducibility

%% Configuration Parameters
Nt = 8;% Number of transmit antennas
Nr = 4;% Number of receive antennas
Nris= 15^2;% Total RIS elements (square array)
K = 1; % Rician K-factor
D= 500; % TX–RX distance [m]
dist_ris= 40; % TX–RIS distance [m]
freq = 2e9;   % Carrier frequency [Hz]
offT = 20;  % TX array offset [m]
offR = 100; % RX array offset [m]
Pt = 1; % Transmit power [W]
noise_dB= -120;% Noise power [dB]
SNR = 10^( -noise_dB/10 );
nTrials = 10;% Number of channel realizations
nIter= 500; % Maximum PGM iterations
alpha_d = 3; % Path-loss exponent for direct link

%% 1. Generate channels using original script
% Outputs: cell arrays Hdirt{t}, H1t{t}, H2t{t}
[Hdirt, H1t, H2t] = chan_mat_RIS_surf_univ_new( ...
    Nt, Nr, Nris, offT, offR, D, nTrials, K, freq, dist_ris, alpha_d);

%% 2. Loop over channel realizations and run PGM optimization
avgRate = zeros(1, nIter+1);
for t = 1:nTrials
    Hdir = Hdirt{t};% Direct channel
    H1 = H1t{t};  % TX→RIS channel
    H2 = H2t{t}; % RIS→RX channel

    % Spectral-norm normalization constant
    c = sqrt(norm(Hdir) / norm(H2*H1)) * max(sqrt(Pt),1) / sqrt(Pt) * 10;

    % Initial transmit covariance and RIS phase vector
    Q0   = (Pt/Nt) * eye(Nt);
    phi0 = ones(Nris,1) / c;

    % Run the fixed PGM solver
    [rateSeq, ~] = optimizePGM(Pt, ...
        Hdir*sqrt(SNR)/c, ...  % Direct link scaled by SNR and c
        H1*sqrt(SNR),   ...    % TX→RIS link scaled by SNR
        H2,             ...    % RIS→RX link
        nIter, Q0*c^2, phi0, c);

    avgRate = avgRate + rateSeq;
end
avgRate = avgRate / nTrials;

%% 3. Plot convergence
figure;
semilogx(0:nIter, avgRate, 'r-', 'LineWidth', 2);
xlabel('Iteration number');
ylabel('Achievable rate [bit/s/Hz]');
title('PGM Convergence Rate');
grid on;

%% Fixed PGM Solver Definition
function [rateHist, elapsed] = optimizePGM(Pt, H0, H1, H2, maxIter, Qstart, phiStart, normC)
    tolVal = 1e-5; % Line-search tolerance
    alphaLS = 0.5;% Backtracking factor
    tau0 = 1e4; % Initial step size

    Q = Qstart;% Transmit covariance
    phi = phiStart;% RIS phase vector

    rateHist = computeRate(H0, H1, H2, phi, Q);
    prevRate = rateHist;
    elapsed = zeros(1, maxIter);
    tStart = tic;

    for k = 1:maxIter
        tau = tau0;  % Reset step size each iteration

        % Compute gradients
        dQ   = gradCov(H0, H1, H2, phi, Q);
        dPhi = gradRIS(H0, H1, H2, phi, Q);

        % Backtracking line search
        for ls = 1:20
            Qcand = projCov(Q + tau*dQ, Pt*normC^2);
            phicand = unitProj(phi + tau*dPhi) / normC;
            rNew = computeRate(H0, H1, H2, phicand, Qcand);

            if (rNew - prevRate >= tolVal * ...
                    (norm(Qcand-Q,'fro')^2 + norm(phicand-phi)^2))
                Q = Qcand;
                phi = phicand;
                prevRate = rNew;
                break;
            else
                tau = tau*alphaLS;
            end
        end

        rateHist(end+1) = rNew;
        elapsed(k)= toc(tStart);
    end
end

%% Supporting Functions: rate, gradients, projection, water-filling
function R = computeRate(Hd, H1, H2, phi, Q)
    M = Hd + H2 * diag(phi) * H1;
    R = real(log2(det(eye(size(M,1)) + M * Q * M')));
end

function G = gradCov(Hd, H1, H2, phi, Q)
    M = Hd + H2 * diag(phi) * H1;
    T = inv(eye(size(M,1)) + M * Q * M');
    G = M' * T * M;
end

function Gphi = gradRIS(Hd, H1, H2, phi, Q)
    M = Hd + H2 * diag(phi) * H1;
    T = inv(eye(size(M,1)) + M * Q * M');
    Gphi = diag(H2' * T * M * Q * H1');
end

function x = unitProj(x)
    x = x ./ abs(x); % Project onto unit magnitude
end

function Qp = projCov(Qm, Pt)
    [V,D] = eig((Qm+Qm')/2);
    d = real(diag(D));
    dHat = waterFill(Pt, d);
    Qp = V * diag(dHat) * V';
end

function p = waterFill(Ptot, eigVals)
    [sv,~] = sort(eigVals,'descend');
    N = length(sv);
    for i = N:-1:1
        lvl = (sum(sv(1:i)) - Ptot) / i;
        if all(sv(1:i) - lvl >= 0), break; end
    end
    p = max(eigVals - lvl, 0);
end

function [Hdir,H1,H2] = chan_mat_RIS_surf_univ_new(Nt,Nr,Nris,lt,lr,D,no_mat,K,f,dist_ris,varargin)

lambda = 3e8/f; % Wavelength
dt = lambda/2; % TX antenna space
dr = lambda/2; % RX antenna space
dris = lambda/2;  % RIS element space
k = 2*pi/lambda;  % Wavenumber


% TX antenna array
tx_arr(1,:) = zeros(1,Nt); 
tx_arr(2,:) = (sort(0:Nt-1,'descend')-(Nt-1)/2)*dt+lt; 
tx_arr(3,:) = zeros(1,Nt); 
% RX antenna array
rx_arr(1,:) = D*ones(1,Nr);
rx_arr(2,:) = (sort(0:Nr-1,'descend')-(Nr-1)/2)*dr+lr; 
rx_arr(3,:) = zeros(1,Nr);
% RIS 
center = [dist_ris 0]; % RIS center position 
N1 = sqrt(Nris);
N2 = N1;  % Number of RIS elements in two dimensions N1 and N2
ris_pos = RISPosition(N1,N2,dris,center);   % RIS elements' coordinates 
a = repmat(ris_pos{1},N1,1); % Placing RIS elements in proper coordinates
ris_arr(1,:) = a(:)';        
ris_arr(2,:) = zeros(1,Nris);
ris_arr(3,:) = repmat(ris_pos{2},1,N2); 

if isempty(varargin)   % Load the FSPL of the direct link
    alpha = 2;
else
    alpha = varargin{1};
end

% direct TX-RX paths/channel matrix
for i1 = 1:Nr  % Distance between the TX and RX antennas                                           
    for j1 = 1:Nt
        d(i1,j1) = norm(rx_arr(:,i1)-tx_arr(:,j1));
    end 
end
Hdir_los = exp(-1i*k*d);    % Direct link, LOS matrix exponents 
tx_rx_dist = sqrt(D^2+(lt-lr)^2);  % TX-RX distance   
FSPL_dir = (lambda/(4*pi))^2/tx_rx_dist^alpha(1);  % Inversion of the FSPL of the direct link 
Hdir = Rician(Hdir_los,sqrt(FSPL_dir),no_mat,K); % Direct link channel matrix           


% indirect paths (TX-RIS-RX)
for l1 = 1:Nris   % Distance between the RIS elements and the RX antennas                                                            
    for r1 = 1:Nr  
        d2(r1,l1) = norm(rx_arr(:,r1)-ris_arr(:,l1)); 
    end
    for t1 = 1:Nt  % Distance between the RIS elements and the TX antennas                  
        d1(l1,t1) = norm(tx_arr(:,t1)-ris_arr(:,l1));   
    end
end

tx_ris_dist = sqrt(dist_ris^2+lt^2);  % TX-RIS distance
ris_rx_dist = sqrt((D-dist_ris)^2+lr^2);  % RIS-RX distance   

FSPLindir = lambda^4/(256*pi^2)*...    % Inversion of the FSPL of the indirect link 
           ((lt/tx_ris_dist+lr/ris_rx_dist)^2)*...
           1/(tx_ris_dist*ris_rx_dist)^2;

% TX-RIS channel
H1_los = exp(-1i*k*d1);  % TX-RIS link, LOS matrix exponents  
FSPL_1 = sqrt(FSPLindir);  % FSPL of the indirect link is embedded in the TX-RIS channel matrix 
H1 = Rician(H1_los,FSPL_1,no_mat,K);

% RIS-RX channel 
H2_los = exp(-1i*k*d2);   % RIS-RX link, LOS matrix exponents 
FSPL_2 = 1;
H2 = Rician(H2_los,FSPL_2,no_mat,K);
end

function pos = RISPosition(N1,N2,dist,center)  % Positions of RIS elements
d1 = (0:N1-1)-(N1-1)/2;
d2 = (0:N2-1)-(N2-1)/2;
pos{1} = center(1)+d1*dist;
pos{2} = center(2)+d2*dist;
end 

function Hout = Rician(Hlos,FSPL,no_mat,K) % Rician channel matices
Hlos = repmat(Hlos,no_mat,1);
Hnlos = sqrt(1/2)*(randn(size(Hlos))+1i*randn(size(Hlos)));
Htot = FSPL/sqrt(K+1)*(Hlos*sqrt(K)+Hnlos);
dim = size(Hlos,1)/no_mat;
for ind = 1:no_mat
   Hout{ind} = Htot((ind-1)*dim+1:ind*dim,:); 
end
end


