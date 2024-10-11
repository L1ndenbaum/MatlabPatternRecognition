% Data generation (random for demonstration purposes)
rng(1); % for reproducibility
data = [randn(50,2)+1; randn(50,2)-1; randn(50,2)+[3,-3]]; % 3 clusters of data

% Number of clusters
k = 3;

% Set up the GIF file
filename = 'clustering_process.gif';
figure;

% Perform k-means clustering and capture frames
[idx, centroids] = kmeans(data, k, 'MaxIter', 10, 'Display', 'iter', 'Start', 'plus');

for iteration = 1:10
    % Recalculate the cluster centroids and reassign data points
    [idx, centroids] = kmeans(data, k, 'MaxIter', iteration, 'Start', centroids, 'Display', 'off');
    
    % Plot the data points and centroids
    gscatter(data(:,1), data(:,2), idx);
    hold on;
    plot(centroids(:,1), centroids(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3); % Plot centroids
    title(['Iteration ', num2str(iteration)]);
    hold off;
    
    % Capture the current figure as an image
    frame = getframe(gcf);
    img = frame2im(frame);
    [imind, cm] = rgb2ind(img, 256);
    
    % Write to the GIF file
    if iteration == 1
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 1);
    else
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 1);
    end
end
