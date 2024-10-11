train_X = [0 0; 2 1; 1 0; -1 1; -2 0; -2 -1; 0 -2; 0 -1; 1 -2];
train_y = [1; 1; 1; 2; 2; 2; 3; 3; 3];
num_classes = size(unique(train_y), 1);
num_features = size(train_X(1, :), 2);
feature_vec = [-2 2];

mu = zeros(num_classes, num_features);
covariance = zeros(num_classes, num_features, num_features);
for i = 1:num_classes
    class_samples = train_X(train_y == i, :); 
    mu(i, :) = mean(class_samples);
    covariance(i, :, :) = cov(class_samples);
end

% 结果
[~, pred_class_of_x] = predict(feature_vec, mu, covariance, num_classes);
disp("x=[-2, 2]的预测类别是:");
disp(pred_class_of_x);

% 绘图
plot_style = {'r', 'g', 'b'};
x_min = min(train_X(:, 1)) -1; 
x_max = max(train_X(:, 1)) +1;
y_min = min(train_X(:, 2)) -1; 
y_max = max(train_X(:, 2)) +1;
[x1Grid, x2Grid] = meshgrid(x_min:0.01:x_max, y_min:0.01:y_max);
grid_points = [x1Grid(:), x2Grid(:)]; 
[boundary1, boundary2, boundary3] = get_decision_boundary(grid_points, mu, covariance, num_classes);
plot(boundary1(:, 1), boundary1(:, 2), boundary2(:, 1), boundary2(:, 2), boundary3(:, 1), boundary3(:, 2))
hold on;
for i = 1:num_classes
     class_samples = train_X(train_y == i, :);
     scatter(class_samples(:, 1), class_samples(:, 2), 'filled', plot_style{i});
     hold on;
end
xlabel('特征1');
ylabel('特征2');
title('结果');
grid on;
hold on;
scatter(feature_vec(1), feature_vec(2), "filled", plot_style{pred_class_of_x})
legend('12类边界', '13类边界', '23类边界', '1类样本点', '2类样本点', '3类样本点', "被预测的X", 'location', 'southeast');

% 打分函数 由于先验概率相同 这里只使用多维正态分布概率作为分数
function score = score_function(feature_vec, class_mean, class_covar)
    score = mvnpdf(feature_vec, class_mean, class_covar);
    score = log(score);
end

% 预测函数
function [score, pred_label] = predict(feature_vec, mu, covariance, num_classes)
    pred_probability = zeros(1, num_classes);
    for i = 1:num_classes
        pred_probability(i) = score_function(feature_vec, mu(i, :), squeeze(covariance(i, :, :)));
    end
    [score, pred_label] = max(pred_probability);
end

function [boundary1, boundary2, boundary3] = get_decision_boundary(grid_points, mu, covariance, num_classes)
    pred_labels = zeros(size(grid_points, 1), 1);
    grid_scores = zeros(size(grid_points, 1), 1);
    boundary1 = []; boundary2 = []; boundary3 = [];
    for i = 1:size(grid_points, 1)
        [grid_scores(i), pred_labels(i)] = predict(grid_points(i, :), mu, covariance, num_classes);
    end

    for i = 1:size(pred_labels, 1)-1
        if abs(grid_scores(i+1)-grid_scores(i)) < 1e-1
            if((pred_labels(i+1)==1 && pred_labels(i)==2) || (pred_labels(i+1)==2 && pred_labels(i)==1))
                boundary1 = [boundary1; (grid_points(i, :)+grid_points(i+1, :))/2;];
            elseif((pred_labels(i+1)==1 && pred_labels(i)==3) || (pred_labels(i+1)==3 && pred_labels(i)==1))
                boundary2 = [boundary2; (grid_points(i, :)+grid_points(i+1, :))/2;];
            elseif((pred_labels(i+1)==2 && pred_labels(i)==3) || (pred_labels(i+1)==3 && pred_labels(i)==2))
                boundary3 = [boundary3; (grid_points(i, :)+grid_points(i+1, :))/2;];
            end
        end
    end
end