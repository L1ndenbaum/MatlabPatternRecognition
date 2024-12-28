classdef Population < handle
    % 代表种群的类
    properties
        chromosome_length, lower_bound, upper_bound, pred_max_value
        population_size, chromosome, best_chromosome % best_chromosome即最佳的x点
        fitness, best_fits, avg_fits
        cross_probability, mutation_probability
        best_fitness_trace, avg_fitness_trace % 记录种群每一代最好的适应度和平均的适应度
    end
    
    methods
        function obj = Population(chromosome_length, population_size, pred_max_value,...
                                  lower_bound, upper_bound, cross_probability, mutation_probability)
            obj.chromosome_length = chromosome_length;
            obj.population_size = population_size;
            obj.chromosome = zeros(population_size, chromosome_length);
            for i = 1:population_size
                obj.chromosome(i, :) = randi([0, 1], 1, chromosome_length);
            end
            obj.fitness = zeros(population_size, 1);
            obj.best_fits = [];
            obj.avg_fits = [];
            obj.cross_probability = cross_probability;
            obj.mutation_probability = mutation_probability;
            obj.best_fitness_trace = [];
            obj.avg_fitness_trace = [];
            obj.lower_bound = lower_bound;
            obj.upper_bound = upper_bound;
            obj.pred_max_value = pred_max_value;
            obj.get_fitness_()
        end
        
        function value = f(~, x)
            value = -x*x - 4*x + 1;
        end

        function encoded_float = encode(obj, chromosome)
            temp = 0;
            for i = obj.chromosome_length:-1:1
                temp = temp + chromosome(i) * power(2, obj.chromosome_length-i);
            end
            encoded_float = ...
            obj.lower_bound + temp * ((obj.upper_bound-obj.lower_bound) / power(2, obj.chromosome_length)-1);
        end
        
        function get_fitness_(obj)
            % 设定个体的适应度为1 / (pred_max_value - f(x))  
            for idx = 1:obj.population_size
                obj.fitness(idx) = 1 / (obj.pred_max_value - obj.f(obj.encode(obj.chromosome(idx, :))));
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
                pos = ceil(rand * obj.chromosome_length);
                while pos == 0
                    pos = ceil(rand * obj.chromosome_length);
                end
                pos = 1 : pos;
                
                % 交叉操作的实现
                temp_chromosome_part = obj.chromosome(index(1), pos);
                obj.chromosome(index(1), pos) = obj.chromosome(index(2), pos);
                obj.chromosome(index(2), pos) = temp_chromosome_part;
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

                % 执行变异操作
                obj.chromosome(index, :) = randi([0, 1], 1, obj.chromosome_length);    
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


