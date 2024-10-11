data = readmatrix("../Data/BayesClassifierData.xlsx");
data(:, 4) = int16(data(: , 4));
n_train = 29;
n_test = 30;
train_X = data(1:n_train, 1:3);
train_y = data(1:n_train, 4);
test_X = data(n_train+1:n_test+n_train, 1:3);

%% 最小错误率贝叶斯分类
% 先验概率
classes = unique(train_y); 
num_classes = 4;
prior = zeros(num_classes, 1);
for i = 1:num_classes
    prior(i, 1) = sum(train_y == classes(i)) / n_train;
end

% 计算每个类别的每个特征的均值和协方差矩阵
mu = zeros(num_classes, 3);
sigma = zeros(num_classes, 3);
covariance = zeros(num_classes, 3, 3);
for i = 1:num_classes
    class_samples = train_X(train_y == classes(i), :);  % 类i的样本
    mu(i, :) = mean(class_samples);  % i类每个特征的均值
    covariance(i, :, :) = cov(class_samples);  % i类的协方差矩阵
end
size(covariance(1, :, :))
% 分类测试数据
result = zeros(n_test, 1);
for i = 1:n_test
     result(i) = classify(test_X(i, :), covariance, prior, mu, num_classes);
end
fprintf('最小错误率分类器预测类别:\n');
%disp(result);

plot_style = {'ro', 'go', 'bo', 'ko'};

subplot(1, 3, 1)
for i = 1:num_classes
    class_samples = train_X(train_y==classes(i), :);
    plot3(class_samples(:, 1), class_samples(:, 2), class_samples(:, 3), plot_style{i})
    hold on;
end
title('Train Labels')
legend();
grid on;

subplot(1, 3, 2)
for i = 1:num_classes
    class_samples = test_X(result==classes(i), :);
    plot3(class_samples(:, 1), class_samples(:, 2), class_samples(:, 3), plot_style{i})
    hold on;
end
title('Min Error Pred Distribution')
legend();
grid on;

%% 最小风险贝叶斯分类
risk_matrix = ones(4) - eye(4); % 风险表
%risk_matrix = rand(4, 4);
for i = 1:4
    risk_matrix(i, i) = 0;
end

for i = 1:n_test
    result(i) = classify_min_fault(test_X(i, :), covariance, prior, mu, num_classes, risk_matrix);
end
fprintf('最小风险分类器预测类别:\n');
disp(result);

subplot(1, 3, 3)
for i = 1:num_classes
    class_samples = test_X(result==classes(i), :);
    plot3(class_samples(:, 1), class_samples(:, 2), class_samples(:, 3), plot_style{i})
    hold on;
end
title('Min Risk Pred Distribution')
legend();
grid on;


% 最小错误率的打分函数
function score = classification_function(i, feature_vec, covariance, mu, prior)
    covariance = squeeze(covariance(i, :, :));
    score = mvnpdf(feature_vec, mu(i, :), covariance);
    score = prior(i) * score;
end

% 风险函数
function risk = risk_function(score_classes, risk_matrix, decision) 
    risk = 0;
    for i = 1:4
        risk = risk_matrix(decision, i) * exp(score_classes(i, 1)) + risk;
    end
end

% 根据打分函数输出预测类别
function pred_class = classify(feature_vec, covariance, prior, mu, num_classes)
    score_classes = zeros(1, num_classes);
    for i = 1:num_classes
       score_classes(i) = classification_function(i, feature_vec, covariance, mu, prior);
    end
    [~, pred_class] = max(score_classes); 
end

function pred_class = classify_min_fault(feature_vec, covariance, prior, mu, num_classes, risk_matrix)
    score_classes = zeros(num_classes, 1);
    class_risk = zeros(4, 1);
    for i = 1:num_classes
       score_classes(i, 1) = classification_function(i, feature_vec, covariance, mu, prior);
    end
    
    for decision = 1:4 
        class_risk(decision) = risk_function(score_classes, risk_matrix, decision);
    end
    [~, pred_class] = min(class_risk);
end
