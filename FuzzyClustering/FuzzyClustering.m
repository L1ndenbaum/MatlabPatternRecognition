%% 数据导入
data = readmatrix("../Data/BayesClassifierData.xlsx");
data(:, 4) = int16(data(: , 4));
n_samples = size(data, 1);  n_train = 29;   n_test = 59 - n_train;
train_X = data(1:n_train, 1:3);
train_y = data(1:n_train, 4);
test_X = data(n_train+1:n_train+n_test, 1:3);
pred_y = zeros(n_test, 1);
fis = readfis('FuzzyClustering.fis');

%% 分类并绘图
for i = 1:n_test
    input = test_X(i, :);
    output = int16(round(evalfis(fis, input)));
    pred_y(i) = output;
    disp(['Input: ', num2str(input), '    -->    ', ' Output: ', num2str(output)]);
end

res_visualization(train_X, train_y, test_X, pred_y)

% Functions
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
        if size(class_samples, 1) == 0
            disp("绘制时该类没有样本"); disp(i);
            continue;
        end
        plot3(class_samples(:, 1), class_samples(:, 2), class_samples(:, 3), plot_styles{i})
        hold on;
    end
    grid on;
    title('Test Distribution')
    legend();
end