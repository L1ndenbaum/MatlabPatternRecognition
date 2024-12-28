%% 数据导入
data = readmatrix("../Data/BayesClassifierData.xlsx");
data(:, 4) = int16(data(: , 4));
num_samples = size(data, 1);
X = data(1:num_samples, 1:3);

%% 超参数定义
num_clusters = 4;
num_ants = 500;
num_iterations = 100;
evaporation_rate = 0.5;

%% 迭代初始化
pheromone = ones(num_clusters, 1);
centers = X(randperm(size(X, 1), num_clusters), :);
best_cluster_centers = centers;
best_cost = inf;

%% 迭代
for iter = 1:num_iterations
    cluster_assignments = zeros(size(X, 1), 1);
    for ant = 1:num_ants
        for i = 1:num_samples
            distances = sqrt(sum((X(i, :) - centers).^2, 2));
            probs = pheromone .^ 2 ./ distances;
            probs = probs / sum(probs);
            temp = find(rand <= cumsum(probs), 1);
            if size(temp, 1) == 0
                r = rand;
                while r==0
                    r = rand;
                end
                temp = ceil(r*num_clusters);
            end
            cluster_assignments(i) = temp;
        end
        
        % 更新中心
        for j = 1:num_clusters
            centers(j, :) = mean(X(cluster_assignments == j, :), 1);
        end
        
        % 计算损失
        cost = 0;
        for i = 1:size(X, 1)
            cost = cost + sum((X(i, :) - centers(cluster_assignments(i), :)).^2);
        end
        
        % 更新信息素
        pheromone = pheromone * evaporation_rate;  % 蒸发信息素
        pheromone(cluster_assignments) = pheromone(cluster_assignments) + 1 / cost;  % 更新信息素
    
    end
    
    % 更新最优解
    if cost < best_cost
        best_cost = cost;
        best_cluster_centers = centers;
    end
    
    disp(['Iteration ' num2str(iter) ', Cost: ' num2str(cost)]);
end

disp('Best Cluster Centers:');
disp(best_cluster_centers);

centers = best_cluster_centers;

%% 分类过程
pred_y = classification(X, centers);
res_visualization(X, pred_y, centers, num_clusters);

%% 函数
function pred_y = classification(test_X, centers)
    n = size(test_X, 1);
    pred_y = zeros(n, 1);
    for i = 1:n
        distances = sum(power(centers-test_X(i, :), 2), 2);
        [~ ,pred_y(i)] = min(distances);
    end
end

function res_visualization(X, pred_y, centers, K)
    plot_styles = {'ro', 'go', 'bo', 'ko'};
    center_styles = {'rx', 'gx', 'bx', 'kx'};
    for i = 1:K
        class_samples = X(pred_y==i, :);
        plot3(centers(i, 1), centers(i, 2), centers(i, 3), center_styles{i}, 'MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot3(class_samples(:, 1), class_samples(:, 2), class_samples(:, 3), plot_styles{i});
    end
    grid on;
    title('Data Distribution');
    legend('Class 1 Center', 'Class 1', 'Class 2 Center', 'Class 2', ...
        'Class 3 Center', 'Class 3', 'Class 4 Center', 'Class 4');
end
