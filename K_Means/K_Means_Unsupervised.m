%% 数据导入
data = readmatrix("../Data/BayesClassifierData.xlsx");
data(:, 4) = int16(data(: , 4));
n_samples = size(data, 1);
X = data(1:n_samples, 1:3);
y = zeros(n_samples, 1);

%% 超参数与初始化
K = 4;
[y, centers] = init(X, y, K);
last_loss = loss_function(X, y, centers, K);

%% 迭代过程
num_iterations = 0;
figure('Position', [100, 100, 1400, 900]);

while true
    num_iterations = num_iterations + 1;

    plot_centers(centers, K);
    [y, centers, loss] = iteration(X, y, K, centers, n_samples);
    plot_res(X, y, K);
    legend('Class 1 Center', 'Class 2 Center', 'Class 3 Center', 'Class 4 Center', ...
       'Class 1', 'Class 2', 'Class 3', 'Class 4');
    title(['Iteration ', num2str(num_iterations)]);
    hold off;

    frame = getframe(gcf);
    img = frame2im(frame);
    [imind, cm] = rgb2ind(img, 256);
    if num_iterations == 1
        imwrite(imind, cm, "ClusteringProcess_Unsupervised.gif", 'gif', 'Loopcount', inf, 'DelayTime', 1.5);
    else
        imwrite(imind, cm, "ClusteringProcess_Unsupervised.gif", 'gif', 'WriteMode', 'append', 'DelayTime', 1.5);
    end
    if loss == last_loss
        break;
    else
        last_loss = loss;
    end
end

res_visualization(X, y, K);

%% 函数
function [y, centers] = init(X, y, K)
    n = size(y, 1);
    center_idxs = randperm(n, K);
    centers = X(center_idxs, :);
    y(center_idxs) = linspace(1, K, K);
end

function [y, centers, loss] = iteration(X, y, K, centers, n_samples)
    for i = 1:n_samples
        distance = sum(power(centers-X(i, :), 2), 2);
        [~, min_idx] = min(distance);
        y(i) = min_idx;
    end

    means = zeros(K, size(X, 2));
    for i = 1:K
        disp(X(y==i, :));
        disp(mean(X(y==i, :)));
        means(i, :) = mean(X(y==i, :));
    end
    centers = means;
    loss = loss_function(X, y, means, K);
end

function loss = loss_function(X, y, means, K)
    loss = 0;
    for i = 1:K
        loss = loss + sum(sum(power(X(y==i|y==0, :) - means(i, :), 2), 1), 2);
    end
end

function plot_centers(centers, K)
    plot_styles = {'rx', 'gx', 'bx', 'kx'};
    for i = 1:K
        plot3(centers(i, 1), centers(i, 2), centers(i, 3), plot_styles{i}, 'MarkerSize', 15, 'LineWidth', 3);
        hold on;
    end
end

function plot_res(X, y, K)
    plot_styles = {'ro', 'go', 'bo', 'ko'};
    for i = 1:K
        class_samples = X(y==i, :);
        if size(class_samples, 1) == 0
            disp("绘制时该类没有样本"); disp(i);
            continue;
        end
        plot3(class_samples(:, 1), class_samples(:, 2), class_samples(:, 3), plot_styles{i})
        hold on;
    end
    grid on;

end

function res_visualization(X, y, K)
    plot_styles = {'ro', 'go', 'bo', 'ko'};
    subplot(1, 2, 1) % 原数据分布
    plot3(X(:, 1), X(:, 2), X(:, 3), 'ro');
    grid on;
    title('Data Distribution')

    subplot(1, 2, 2) % 聚类后数据
    for i = 1:K
        class_samples = X(y==i, :);
        if size(class_samples, 1) == 0
            disp("绘制时该类没有样本"); disp(i);
            continue;
        end
        plot3(class_samples(:, 1), class_samples(:, 2), class_samples(:, 3), plot_styles{i})
        hold on;
    end
    grid on;
    title('Clusters Distribution')
    legend();
end