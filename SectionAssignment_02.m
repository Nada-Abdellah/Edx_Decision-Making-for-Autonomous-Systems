clc;

ml = 4; % Memory length, must be greater than 0
as = 2; % Action space size, must be greater than 1
ss = as^ml; % State space
            
t = 1;
x1(:,t) = [1 1 1 1]';
x2(:,t) = [1 1 1 1]';

alpha = 0.25;
gamma = 0.95;
vareps = 1;

score1 = 0;
score2 = 0;
maxscore = 1;

epilen = 200;
rwMver = 2;

rwM = generateRwM(rwMver, maxscore);
statM1 = zeros(ss,as);
statM2 = zeros(ss,as);

history1 = "";
history2 = "";

% ------------------------------------------------------------------
% Main loop
while t < epilen+1
       
    ssnr1 = getStateNumber(as, ml, x1(:,t));
    ssnr2 = getStateNumber(as, ml, x2(:,t));
    
    u1 = simpleDecisionAlgorithm(as, ml, x1(:,t), statM1);
    u2 = QlearningDecision(as, ml, x2(:,t), vareps, statM2);
     
    y1 = getReward(as, ml, x1(:,t), u1, rwM, 0);
    y2 = getReward(as, ml, x2(:,t), u2, rwM, 0);
    
    score1 = score1 + y1;
    score2 = score2 + y2;
    
    if mod(t,10) == 0
        crchar = '\n';
    else
        crchar = "";
    end
    
    history1 = strcat(history1,getActionChar(u1),":",num2str(y1)," | ", crchar);
    history2 = strcat(history2,getActionChar(u2),":",num2str(y2)," | ", crchar);
   
    x1(:,t+1) = updateState(as, x1(:,t), u1, 0);
    x2(:,t+1) = updateState(as, x2(:,t), u2, 0);
    
    statM1 = simpleUpdateStatistics(as, ml, x1(:,t), y1, u1, statM1);
    statM2 = QlearningUpdateMatrix(as, ml, x2(:,t), x2(:,t+1), y2, u2, alpha, gamma, statM2);
    
    t = t + 1;

end

% ------------------------------------------------------------------
% Output episode results

fprintf(strcat("Length of episode: ",num2str(epilen),'\n','\n'))

fprintf(strcat("Simple algorithm actions:",'\n', history1));
fprintf(strcat("Simple algorithm score: ",num2str(score1)," points",'\n'))

fprintf(strcat('\n', "Q-learning algorithm actions:",'\n', history2'));
fprintf(strcat("Q-learning algorithm score: ",num2str(score2)," points",'\n'))

fprintf(strcat('\n', "Q-matrix at end of episode:",'\n'))
disp(statM2)

% ------------------------------------------------------------------

% Calculates the corresponding state number from the state vector
function ssnr = getStateNumber(as, ml, x)
    baseWeights = flip(as.^(0:(ml-1)));
    ssnr = sum((x'-1).*baseWeights)+1;
end
        
% Translates the action number to the corresponding character symbol 
% 1 -> "A", 2 -> "B", 3 -> "C", 4 -> "D"
function actionChar = getActionChar(anr)
    actionChar = char(anr+64);
end
 
% Translates a state vector to a character symbol string 
function stateString = getStateString(x)
    stateString = [];
    [n,m]=size(x);
    for i=1:n
        actionChar = getActionChar(x(i,1));
        stateString = strcat(stateString, actionChar);
    end
end

% Calculates the corresponding state vector from the state number
function stateVector = getStateVector(as, ml, ssnr)
    ssnr = ssnr - 1;
    stateVector = zeros(ml,1);
    i = ml;
    while i > 0
        bweight = power(as,(i-1));
        stateVector(ml-i+1) = floor(ssnr/bweight);
        ssnr = ssnr - stateVector(ml-i+1)*bweight;
        i = i - 1;
    end
    stateVector = stateVector + 1;
end

function newState = updateState(as, x, u, prb)
    newState = circshift(x, 1);
    if rand < prb
        uvect = 1:as;
        uvect(uvect==u)=[];
        u = uvect(randi(length(uvect)));      
    end
    newState(1) = u;
end

function reward = getReward(as, ml, x, u, rwM, prb)
    ssnr = getStateNumber(as, ml, x);
    reward = rwM(ssnr,u);
    if rand < prb
        reward = -1*reward;
    end  
end

function rwM = generateRwM(rMver, maxscore)
    
    if rMver == 1
        rwM = -1*ones(16,2);
        rwM(1,2) = 1;
        rwM(:,1) = [-1 1 1 0 1 0 0 0 1 0 0 0 0 0 0 0]';
    elseif rMver == 2
        rwM = zeros(16,2);
        rwM(1,1) = -1;
        rwM(15,2) = 1;
        rwM(16,1) = 1;
        rwM(16,2) = -1;
    elseif rMver == 3
        rwM = zeros(16,2);
        rwM(3,2) = 1;
        rwM(5,1) = 1;
        rwM(10,1) = 1;
        rwM(3,1) = -1;
        rwM(5,2) = -1;
        rwM(10,2) = -1;       
        rwM(8,2) = maxscore;
        rwM(12,1) = maxscore;
        rwM(6,2) = maxscore;
        rwM(11,2) = maxscore;    
        rwM(15,2) = maxscore;
        rwM(16,1) = maxscore;
        rwM(16,2) = -1;
    elseif rMver == 4
        rwM = zeros(16,2);
        rwM(16,1) = maxscore;
    end
    
end

% Function for updating a simple algorithm statistics matrix
function statistics = simpleUpdateStatistics(as, ml, x, y, u, statistics)
    ssnr = getStateNumber(as, ml, x);
    statistics(ssnr, u) = statistics(ssnr,u) + y;
end

% Function for a simple algorithm using a statistics matrix
function u = simpleDecisionAlgorithm(as, ml, x, statistics)
    ssnr = getStateNumber(as, ml, x);
    
    if statistics(ssnr,as) == 2
        u = 1; % Taking action "A"
    elseif statistics (ssnr,as) == 1
        u = 2; % Taking action "B"
    else
        u = randi(2); % Taking a random action
    end
    
end

% Function for taking a decision automatically by a Q-learning algorithm
function u = QlearningDecision(as, ml, x, vareps, Qmatrix)
    ssnr = getStateNumber(as, ml, x);
       
    if rand > vareps
        if Qmatrix(ssnr, as) == 2
            u = 1;
        elseif Qmatrix(ssnr,as) == 1
            u = 2;
        else
            u = randi(as);
        end
    else
        u = randi(as); % Exploration according to vareps
    end
end
        
% Function for updating the matrix for a Q-learning algorithm
function Qmatrix = QlearningUpdateMatrix(as, ml, x, x_new, y, u, alpha, gamma, Qmatrix)
    ssnr = getStateNumber(as, ml, x);
    ssnr_new = getStateNumber(as, ml, x_new);
           
    Qmatrix(ssnr_new, u) = Qmatrix(ssnr, u) + alpha * (y + gamma * max(Qmatrix(ssnr_new, u)) - Qmatrix(ssnr,u));
                
end