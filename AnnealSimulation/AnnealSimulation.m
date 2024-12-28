%% 数据导入
data = readmatrix("../Data/BayesClassifierData.xlsx");
data(:, 4) = int16(data(: , 4));
num_samples = size(data, 1);
X = data(1:num_samples, 1:3);
y = data(1:num_samples, 4);
num_features = size(X, 2);
num_classes = size(unique(y), 1);

%% 超参数
T_original = 10; % 初始温度
T_end = 0.1; % 结束时温度
num_clusters = 4; % 聚类的簇数
cooling_rate = 0.9;
num_disturbs = 100;

%% 迭代
[current_labels, pred_centers, loss_trace, num_epochs] = iteration(num_clusters, X, ...
                                                   T_original, T_end, num_disturbs, cooling_rate);
res_visualization(X, current_labels, pred_centers, loss_trace, num_epochs, num_clusters);


%% Functions
function current_labels = init_lables(X)
    num_samples = size(X, 1);
    current_labels = zeros(num_samples, 1);

    for i = 1:num_samples
        current_labels(i) = ceil(4*rand);
    end
end

function [current_labels, current_centers, loss_trace, num_epochs] = ...
        iteration(num_clusters, X, ...
                  T_original, T_end, num_disturbs, cooling_rate)
    [num_samples, num_features] = size(X);
    last_labels = init_lables(X);
    last_centers = get_centers(X, last_labels, num_clusters);
    last_loss = loss_function(X, last_labels, num_clusters);
    loss_trace = [];
    current_labels = last_labels;

    T_current = T_original;
    epoch = 0;
    while T_current > T_end
        for i = 1:num_disturbs
            rand_idx = ceil(rand*num_samples); % 随机选择一个样本
            disturb = ceil(rand*(num_clusters)); % 随机产生一个1到num_clusters的标签给这个样本(随机扰动)

            current_labels(rand_idx) = disturb;
            current_centers = get_centers(X, current_labels, num_clusters);
            current_loss = loss_function(X, current_labels, num_clusters);

            % 判断是否接受新解
            if current_loss <= last_loss % 损失降低，接受新解
                last_loss = current_loss;
                last_centers = current_centers;
                last_labels = current_labels;
            else % 损失未降低， 以Metropolis准则接受新解
                if(rand < exp(-(current_loss - last_loss) / T_current))
                    last_loss = current_loss;
                    last_centers = current_centers;
                    last_labels = current_labels;
                else
                    current_loss = last_loss;
                    current_centers = last_centers;
                    current_labels = last_labels;
                end
            end
            epoch = epoch + 1;
            loss_trace = [loss_trace, current_loss];
        end

        % 扰动过程结束，降温
        T_current = cooling_function(T_current, cooling_rate);
    end
    num_epochs = epoch;
end

function T_current = cooling_function(T_current, cooling_rate)
    T_current = T_current * cooling_rate;
end

function loss = loss_function(X, y, num_clusters) % 计算以当下的y作为标签, 总体X的损失
    centers = get_centers(X, y, num_clusters);
    loss = 0;
    for i = 1:num_clusters
        X_class = X(y==i, :);
        loss = loss + sum(sqrt(sum(power(X_class - centers(i, :), 2), 1)));
    end
end

function centers = get_centers(X, y, num_clusters)
    num_features = size(X, 2);
    centers = zeros(num_clusters, num_features);
    for i = 1:num_clusters
        X_class = X(y==i, :);
        centers(i, :) = mean(X_class, 1);
    end
end

function res_visualization(X, pred_y, pred_centers, loss_trace, num_epochs, num_clusters)
    plot_styles = {'ro', 'go', 'bo', 'ko'};
    center_styles = {'rx', 'gx', 'bx', 'kx'};
    subplot(1, 2, 1) % 聚类结果
    for class = 1:num_clusters
        class_X = X(class==pred_y, :);
        plot3(class_X(:, 1), ...
              class_X(:, 2), ...
              class_X(:, 3), ...
              plot_styles{class});
        hold on;
        plot3(pred_centers(class, 1), pred_centers(class, 2), pred_centers(class, 3), ...
              center_styles{class}, ...
              'MarkerSize', 15, 'LineWidth', 3);
        hold on;
    end
    grid on;
    title('Pred Distribution')
    legend('Class 1', 'Center 1', 'Class 2', 'Center 2', ...
           'Class 3', 'Center 3', 'Class 4', 'Center 4');

    subplot(1, 2, 2) % 损失曲线
    plot(linspace(1, num_epochs, num_epochs), loss_trace, 'r*')
    title('Loss Curve')
    legend('Loss');
end