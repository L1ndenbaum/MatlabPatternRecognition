data = readmatrix("./Data/BayesClassifierData.xlsx");
data(:, 4) = int16(data(: , 4));
n_train = 29;
n_test = 30;
train_X = data(1:n_train, 1:3);
train_y = data(1:n_train, 4);
test_X = data(n_train+1:n_test+n_train, 1:3);
num_classes = 4; num_features = 3;
classes = unique(train_y); 

%之后从13类中分出1和3 从24类中分出2和4

%% 先分13类还是24类
merged_train_y = train_y;
merged_train_y(merged_train_y == 3) = 1; %合并13类 标签为1
merged_train_y(merged_train_y == 4) = 2; %合并24类 标签为2

% 分出13类和24类的阈值点y_0和投影向量
[y_threshold1, w_star1] = binary_classification(train_X, merged_train_y);

% 分出1类和3类的阈值点y_0和投影向量
[y_threshold2, w_star2] = binary_classification(train_X((train_y==1|train_y==3), :), train_y(train_y==1|train_y==3));

% 分出2类和4类的阈值点y_0和投影向量
[y_threshold3, w_star3] = binary_classification(train_X((train_y==2|train_y==4), :), train_y(train_y==2|train_y==4));

pred_labels = predict(test_X, y_threshold1, y_threshold2, y_threshold3, w_star1, w_star2, w_star3);
res_visualization(train_X, train_y, test_X, pred_labels);

function [y_threshold, w_star] = binary_classification(X, y)
    % 二元分类函数 X是两类的所有样本, y是样本标签, classi_label是类i的标签
    num_features = 3; classes = unique(y); n_samples = size(y, 1);
    mu = zeros(2, num_features); % 均值向量
    dispersion_matrix = zeros(2, num_features, num_features); % 离散度矩阵

    for i = 1:2
        class_samples = X(y == classes(i), :);
        mu(i, :) = mean(class_samples);
        S_i = zeros(num_features, num_features);
        for j = 1:size(class_samples, 1)
            diff = class_samples(j, :) - mu(i, :);
            S_i = S_i + (diff' * diff);
        end
        dispersion_matrix(i, :, :) = S_i;
    end

    sum_dispersion_matrix = zeros(num_features, num_features);
    for i = 1:2
        sum_dispersion_matrix = sum_dispersion_matrix + squeeze(dispersion_matrix(i, :, :));
    end

    w_star = inv(sum_dispersion_matrix) * (mu(1, :) - mu(2, :))'; % 最佳投影方向
    y_projection = zeros(n_samples, 1);

    for i = 1:n_samples
        y_projection(i) = X(i, :) * w_star ; % 投影点
    end

    mu_hat = zeros(2, 1); % 两类样本在一维上投影的均值
    n_class_samples = zeros(2, 1);

    for i =1:2
        class_projection = y_projection(y == classes(i));
        n_class_samples(i) = size(class_projection, 1);
        mu_hat(i) = sum(class_projection) / n_class_samples(i);
    end

    y_threshold = (n_class_samples(1)*mu_hat(1) + n_class_samples(2)*mu_hat(2)) / sum(n_class_samples);
end

function pred_class = classification(y_projection, y_threshold, class_label1, class_label2)
    % 根据阈值y_0进行分类的函数
    if y_projection >= y_threshold
        pred_class = class_label1;
    else
        pred_class = class_label2;
    end
end

function pred_labels = predict(test_X, y_threshold1, y_threshold2, y_threshold3, w_star1, w_star2, w_star3)
    % 使用测试集进行预测
    n_samples = size(test_X, 1);
    pred_labels = zeros(n_samples, 1);
    for i = 1:n_samples
        y_projection1 = test_X(i, :) * w_star1; % 一维投影点
        pred_label = classification(y_projection1, y_threshold1, 1, 2);
        if pred_label == 1 % 13类
            y_projection2 = test_X(i, :) * w_star2;
            pred_label = classification(y_projection2, y_threshold2, 1, 3);
        else % 24类
            y_projection3 = test_X(i, :) * w_star3;
            pred_label = classification(y_projection3, y_threshold3, 2, 4);
        end
        pred_labels(i) = pred_label;
    end
end

function res_visualization(train_X, train_labels, test_X, pred_labels)
    plot_styles = {'ro', 'go', 'bo', 'ko'};
    classes = unique(train_labels);
    num_classes = size(classes, 1);
    subplot(1, 2, 1) % 训练数据可视化
    for i = 1:num_classes
        class_samples = train_X(train_labels==classes(i), :);
        plot3(class_samples(:, 1), class_samples(:, 2), class_samples(:, 3), plot_styles{i})
        hold on;
    end
    grid on;
    title('Train Distribution')
    legend();

    subplot(1, 2, 2) % 测试数据可视化
    for i = 1:num_classes
        class_samples = test_X(pred_labels==classes(i), :);
        plot3(class_samples(:, 1), class_samples(:, 2), class_samples(:, 3), plot_styles{i})
        hold on;
    end
    grid on;
    title('Test Distribution')
    legend();
end