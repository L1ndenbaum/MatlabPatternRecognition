classdef Population < handle
    % 代表种群的类
    % 一个种群有population_size条染色体, 每个染色体长12, 代表4类的聚类中心, 每3个数字作为一段表示个中心
    properties
        num_features, num_classes, num_samples
        population_size, chromosome, best_chromosome % best_chromosome即最佳的聚类中心
        fitness, best_fits, avg_fits
        X, y, class_data
        cross_probability, mutation_probability
        best_fitness_trace, avg_fitness_trace % 记录种群每一代最好的适应度和平均的适应度
    end
    
    methods
        function obj = Population(num_features, num_classes, population_size, X, y, ...
                                  cross_probability, mutation_probability)
            obj.num_features = num_features;
            obj.num_classes = num_classes;
            obj.population_size = population_size;
            obj.chromosome = 4000 * rand(population_size, num_classes*num_features);
            obj.fitness = zeros(population_size, 1);
            obj.best_fits = [];
            obj.avg_fits = [];
            obj.X = X;
            obj.y = y;
            obj.num_samples = size(X, 1);
            obj.cross_probability = cross_probability;
            obj.mutation_probability = mutation_probability;
            obj.class_data = {};
            for class = 1:num_classes
                class_data_ = X(y==class, :);
                obj.class_data = [obj.class_data, class_data_];
            end
            obj.best_fitness_trace = [];
            obj.avg_fitness_trace = [];
            obj.get_fitness_()
        end
        
        function get_fitness_(obj)
            for idx = 1:obj.population_size
                distances = zeros(obj.num_classes, 1);
                for class = 1:obj.num_classes
                    distances(class) = sum(sum(sqrt(power(obj.class_data{class} - obj.chromosome(idx, class*3-2:class*3), 2)), 1), 2);
                end
                obj.fitness(idx) = 1 / (sum(distances) / obj.num_classes);
            end

            [best_fitness, best_index] = max(obj.fitness);
            obj.best_chromosome = obj.chromosome(best_index, :);
            avg_fitness = sum(obj.fitness) / obj.population_size;
            obj.best_fitness_trace = [obj.best_fitness_trace, best_fitness];
            obj.avg_fitness_trace = [obj.avg_fitness_trace, avg_fitness];
            
        end

        function operation_crossover_(obj)
            % 交叉操作
            for i = 1:obj.population_size
                % 生成随机数决定是否交叉
                pick = rand;
                while pick == 0
                    pick = rand;
                end

                if pick > obj.cross_probability
                    continue;
                end
                
                % 随机选择交叉个体
                index = ceil(rand(1,2) .* obj.population_size); % 生成随机数表示交叉的两个个体
                while (index(1)==index(2)) || (index(1)*index(2)==0)
                    index = ceil(rand(1,2) .* obj.population_size);
                end
                
                % 随机选择交叉位置
                segment = ceil(rand * 4); % 从染色体的4段中随机选择一个交叉段
                while segment == 0
                    segment = ceil(rand * 4);
                end
                pos = segment * 3 - 2 : segment * 3;
                
                % 交叉操作的实现
                temp = obj.chromosome(index(1), pos);
                obj.chromosome(index(1), pos) = obj.chromosome(index(2), pos);
                obj.chromosome(index(2), pos) = temp;
            end
        end

        function operation_mutation_(obj)
            for i = 1:obj.population_size
                % 由变异概率决定该轮循环是否进行变异
                pick=rand;
                if pick > obj.mutation_probability
                    continue;
                end
                
                % 随机选择进行变异的个体
                pick = rand;
                while pick == 0
                    pick = rand;
                end
                index = ceil(pick * obj.population_size);    
                
                % 随机选择变异位置
                segment = ceil(rand * 4); % 从染色体的4段中随机选择一个变异段
                while segment == 0
                    segment = ceil(rand * 4);
                end
                pos = segment * 3 - 2 : segment * 3;

                obj.chromosome(index, pos) = rand * 4000;    
            end
        end

        function operation_selection_(obj)
            sum_fitness = sum(obj.fitness);
            normalized_fitness = (obj.fitness) ./ sum_fitness;
            index = []; % 被选择的population_size个个体
            
            for i = 1:obj.population_size % 轮盘赌poplulation_size次进行选择
                pick = rand;
                while pick == 0    
                    pick = rand;        
                end

                for j = 1:obj.population_size
                    pick = pick - normalized_fitness(j);    
                    if pick < 0 % 选中
                        index = [index, j];            
                        break;
                    end
                end
            end

            obj.chromosome = obj.chromosome(index, :);
            obj.get_fitness_();
        end
    end
end

