data = readmatrix("./Data/BayesClassifierData.xlsx");
data(:, 4) = int16(data(: , 4));
n_train = 29;
n_test = 59 - n_train;

train_X = data(1:n_train, 1:3);
train_X(:, 4) = 1;

test_X = data(n_train+1:n_test+n_train, 1:3);
test_X(:, 4) = 1;

train_y = data(1:n_train, 4);
num_classes = 4; num_features = 3;
classes = unique(train_y);

r_i = zeros(size(train_X, 1), num_classes);
for i = 1:size(train_X, 1)
    for class = 1:num_classes
        if class == train_y(i)
            r_i(i, class) = 1;
        end
    end
end
%W = 10*rand(num_classes, num_features);
W = [9.0488 2.5806 6.0284;
     9.7975 4.0872 7.1122;
     4.3887 5.9490 2.2175;
     1.1112 2.6221 1.1742];
W(:, num_features+1) = 1;

lr = 0.25;
W = train(lr, W, train_X/norm(train_X), train_y, r_i, num_classes);
pred_labels = predict(W, test_X/norm(test_X));
res_visualization(train_X, train_y, test_X, pred_labels)

function W = train(lr, W, train_X, train_y, r_i, num_classes)
    n_samples = size(train_X, 1);
    d = zeros(n_samples, num_classes);
    while(true)
        % 计算d_i
        for idx = 1:n_samples
            for i = 1:num_classes
                d(idx, i) = W(i, :) * train_X(idx, :)';
            end
        end
        
        % 检查迭代停止的条件
        max_idxs = zeros(n_samples, 1);
        for idx = 1:n_samples
            [~, max_idxs(idx)] = max(d(idx, :), [], 2);
        end
        if n_samples - sum(max_idxs == train_y) <= 2
            break;
        end
        
        % 迭代方程
        for class = 1:num_classes
            for idx = 1:n_samples
                W(class, :) = W(class, :) + train_X(idx, :)*lr*(r_i(idx, class) - train_X(idx, :)*W(class, :)');
            end
        end
    end
end

function pred_class = classification(feature_vec, W)
    classes_scores = W * feature_vec;
    [~, pred_class] = max(classes_scores);
end

function pred_labels = predict(W, test_X)
    n_samples = size(test_X, 1);
    pred_labels = zeros(n_samples, 1);
    for i = 1:n_samples
        pred_labels(i) = classification(test_X(i, :)', W);
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