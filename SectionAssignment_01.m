clc;

ml = 4; % Memory length, must be greater than 0
as = 2; % Action space size, must be greater than 1
ss = as^ml; % State space
            
t = 1;
x1(:,t) = [1 1 1 1]';
x2(:,t) = [1 1 1 1]';

score1 = 0;
score2 = 0;

epilen = 40;
rwMver = 2;

rwM = generateRwM(rwMver, epilen);
statM = zeros(ss,as);

history1 = "";
history2 = "";

% Main loop
while t < epilen+1
       
    ssnr1 = getStateNumber(as, ml, x1(:,t));
    ssnr2 = getStateNumber(as, ml, x2(:,t));
    
    u1 = randomAutoDecision(as);
    u2 = simpleDecisionAlgorithm(as, ml, x2(:,t), statM);
     
    y1 = getReward(as, ml, x1(:,t), u1, rwM, 0);
    y2 = getReward(as, ml, x2(:,t), u2, rwM, 0);
    
    statM = naiveUpdateStatistics(as, ml, x2(:,t), y2, u2, statM);
    
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
    
    t = t + 1;

end

fprintf(strcat("Length of episode: ",num2str(epilen),'\n','\n'))

fprintf(strcat("Random algorithm actions:",'\n', history1));
fprintf(strcat("Random algorithm score: ",num2str(score1)," points",'\n'))

fprintf(strcat('\n', "Simple algorithm actions:",'\n', history2'));
fprintf(strcat("Simple algorithm score: ",num2str(score2)," points",'\n','\n'))

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

% Function for taking a decision automatically by a "random" algorithm
% u(t) = g( u(t) )
function u = randomAutoDecision(as)
    u = randi(as); % Pick a ranom action
end

function rwM = generateRwM(rMver, epilen)
    
    if rMver == 1
        rwM = -1*ones(16,2);
        rwM(1,2) = 0;
        rwM(2,2) = 1;
        rwM(:,1) = [-1 0 1 0 1 0 0 0 1 0 0 0 0 0 0 epilen]';
    elseif rMver == 2
        rwM = -1*ones(16,2);
        rwM(1,2) = 0;
        rwM(2,2) = 1;
        rwM(5,2) = 1;
        rwM(:,1) = [-1 0 1 0 -1 1 0 0 1 0 1 0 0 0 0 epilen]';
    elseif rMver == 3
        rwM = -1*ones(16,2);
        rwM(1,2) = 1;
        rwM(:,1) = [-1 1 1 0 1 0 0 0 1 0 0 0 0 0 0 epilen]';
    end
    
end

function statistics = naiveUpdateStatistics(as, ml, x, y, u, statistics)
    ssnr = getStateNumber(as, ml, x);
    statistics(ssnr, u) = statistics(ssnr,u) + y;
end

% Function for a simple algorithm using a statistics matrix
% u(t) = g( x(t), statistics(t) )
function u = simpleDecisionAlgorithm(as, ml, x, statistics)
    ssnr = getStateNumber(as, ml, x);
    
    if statistics(ssnr) == 1
        
        u = 1; % Taking action "A"
        
    elseif statistics(ssnr) == 1
        
        u = 2; % Taking action "B"
        
    else
        
        u = randi(2); % Taking a random action
    end
    
end