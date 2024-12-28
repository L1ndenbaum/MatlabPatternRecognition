%% 数据导入
data = readmatrix("../Data/BayesClassifierData.xlsx");
data(:, 4) = int16(data(: , 4));
n_samples = size(data, 1);  n_train = 29;   n_test = 59 - n_train;
train_X = data(1:n_train, 1:3);
train_y = categorical(data(1:n_train, 4));
test_X = data(n_train+1:n_train+n_test, 1:3);

%% 网络
layers1 = [
        featureInputLayer(3)
        fullyConnectedLayer(8)
        reluLayer()
        fullyConnectedLayer(16)
        reluLayer()
        fullyConnectedLayer(4)
        softmaxLayer()
        classificationLayer()
        ];

layers2 = [
        featureInputLayer(3)
        fullyConnectedLayer(8)
        sigmoidLayer()
        fullyConnectedLayer(16)
        sigmoidLayer()
        fullyConnectedLayer(4)
        softmaxLayer()
        classificationLayer()
        ];

options1 = trainingOptions('adam', 'MaxEpochs', 48, 'MiniBatchSize', 6, 'Shuffle', 'every-epoch', ...
            'Plots', 'training-progress', 'InitialLearnRate', 0.0025);
options2 = trainingOptions('rmsprop', 'MaxEpochs', 48, 'MiniBatchSize', 6, 'Shuffle', 'every-epoch', ...
            'Plots', 'training-progress', 'InitialLearnRate', 0.01);
net1 = trainNetwork(train_X, train_y, layers1, options1);
net2 = trainNetwork(train_X, train_y, layers2, options2);
net3 = trainNetwork(train_X, train_y, layers1, options2);
pred_y1 = classify(net1, test_X);
res_visualization(train_X, train_y, test_X, pred_y1);
pred_y2 = classify(net2, test_X);
res_visualization(train_X, train_y, test_X, pred_y2);
pred_y3 = classify(net3, test_X);
res_visualization(train_X, train_y, test_X, pred_y3);

%% 结果绘图
function res_visualization(train_X, train_y, test_X, test_y)
    figure()
    test_y = int16(test_y(: , 1)); train_y = int16(train_y(: , 1)); 
    plot_styles = {'ro', 'go', 'bo', 'ko'};
    hold on;
    subplot(1, 2, 1)
    for i = 1:4
        class_samples = train_X(train_y==i, :);
        plot3(class_samples(:, 1), class_samples(:, 2), class_samples(:, 3), plot_styles{i});
        hold on;
    end
    grid on;
    title('Train Distribution');
    legend('Class 1', 'Class 2', 'Class 3', 'Class 4');

    subplot(1, 2, 2)
    for i = 1:4
        class_samples = test_X(test_y==i, :);
        if size(class_samples, 1) == 0
            disp("绘制时该类没有样本"); disp(i);
            continue;
        end
        plot3(class_samples(:, 1), class_samples(:, 2), class_samples(:, 3), plot_styles{i})
        hold on;
    end
    grid on;
    title('Pred Distribution')
    legend();
end