%% 数据导入
data = readmatrix("../../Data/BayesClassifierData.xlsx");
data(:, 4) = int16(data(: , 4));
num_samples = size(data, 1);
X = data(1:num_samples, 1:3);
y = data(1:num_samples, 4);
num_features = size(X, 2);
num_classes = size(unique(y), 1);

%% 超参数
cross_probability = 0.9; % 交叉概率
mutation_probability = 0.01; % 变异概率
population_size = 3000; % 种群大小
num_epochs = 200; % 迭代次数

%% 初始化种群
population = Population(num_features, num_classes, population_size, X, y, ...
                        cross_probability, mutation_probability);

for epoch = 1:num_epochs
    population.operation_selection_()
    population.operation_crossover_()
    population.operation_mutation_()
end

[pred_y, pred_centers] = clustering(population);
res_visualization(population, pred_y, pred_centers, num_epochs);
disp(['正确率:', num2str(sum(pred_y==y, 1) / num_samples * 100), '%']);


%% Functions
function [pred_y, centers] = clustering(population)
    num_features = population.num_features;
    num_classes = population.num_classes;
    num_samples = population.num_samples;
    X = population.X;

    % 取得聚类中心
    centers = zeros(num_classes, num_features);
    for class = 1:num_classes
        centers(class, :) = population.best_chromosome(class*num_features-2:class*num_features);
    end

    % 聚类
    pred_y = zeros(num_samples, 1);
    for idx = 1:num_samples
        distances = zeros(num_classes, 1);
        for class = 1:num_classes
            distances(class) = sum(sqrt(power(X(idx, :) - centers(class, :), 2)), 2);
        end
        [~, min_idx] = min(distances);
        pred_y(idx) = min_idx;
    end
end

function res_visualization(population, pred_y, pred_centers, num_epochs)
    plot_styles = {'ro', 'go', 'bo', 'ko'};
    center_styles = {'rx', 'gx', 'bx', 'kx'};
    figure(1) % 第一张图是聚类分布图
    subplot(1, 2, 1)
    for class = 1:population.num_classes
        class_X = population.class_data{class};
        plot3(class_X(:, 1), ...
              class_X(:, 2), ...
              class_X(:, 3), ...
              plot_styles{class});
        hold on;
        plot3(sum(class_X(:, 1), 1) / size(class_X, 1), ...
              sum(class_X(:, 2), 1) / size(class_X, 1), ...
              sum(class_X(:, 3), 1) / size(class_X, 1), ...
              center_styles{class}, ...
              'MarkerSize', 15, 'LineWidth', 3);
        hold on;
    end
    title('True Distribution')
    grid on;
    legend('Class 1', 'Center 1', 'Class 2', 'Center 2', ...
           'Class 3', 'Center 3', 'Class 4', 'Center 4');

    subplot(1, 2, 2)
    for class = 1:population.num_classes
        class_X = population.X(pred_y==class, :);
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

    figure(2) % 第二张图是遗传迭代时适应度变化图
    plot(linspace(1, num_epochs, num_epochs), population.best_fitness_trace(1:num_epochs), 'r*');
    hold on;
    plot(linspace(1, num_epochs, num_epochs), population.avg_fitness_trace(1:num_epochs), 'b*');
    grid on;
    legend('种群最佳适应度曲线', '种群平均适应度曲线');
end
