%% 超参数
cross_probability = 0.9; % 交叉概率
mutation_probability = 0.01; % 变异概率
population_size = 1000; % 种群大小
num_epochs = 50; % 迭代次数

%% 初始化种群
population = Population(9, population_size, 6, -2, 2, cross_probability, mutation_probability);

for epoch = 1:num_epochs
    population.operation_selection_()
    population.operation_crossover_()
    population.operation_mutation_()
end

true_max_value = f(-2); % f(x)在[-2, 2]的最大值 (对称轴为-2)
pred_max_value = f(population.encode(population.best_chromosome));
error = abs(true_max_value-pred_max_value);
disp(['预测最大值为:', num2str(pred_max_value)])
disp(['预测最大值和真实最大值的误差为:', num2str(error)]);
res_visualization(population, num_epochs);

%% Functions
function value = f(x)
    value = -x*x - 4*x +1;
end

function res_visualization(population, num_epochs)
    figure() % 遗传迭代时适应度变化图
    plot(linspace(1, num_epochs, num_epochs), population.best_fitness_trace(1:num_epochs), 'r*');
    hold on;
    plot(linspace(1, num_epochs, num_epochs), population.avg_fitness_trace(1:num_epochs), 'b*');
    grid on;
    legend('种群最佳适应度曲线', '种群平均适应度曲线');
end