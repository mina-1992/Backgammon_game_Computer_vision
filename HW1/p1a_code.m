%% Your code goes in this file. % Read the image
img = imread('p1_image1.png');

% Convert the image to HSV color space
hsv = rgb2hsv(img);

% Define the color ranges for white and red pieces in the HSV color space
white_range = [0 1 0 0.4 0.5 1];
red_range = [0.96 1 0.5 1 0.5 1];

% Threshold the image based on the color ranges
white_mask = (hsv(:,:,1)>=white_range(1) & hsv(:,:,1)<=white_range(2) ...
    & hsv(:,:,2)>=white_range(3) & hsv(:,:,2)<=white_range(4) ...
    & hsv(:,:,3)>=white_range(5) & hsv(:,:,3)<=white_range(6));
red_mask = (hsv(:,:,1)>=red_range(1) & hsv(:,:,1)<=red_range(2) ...
    & hsv(:,:,2)>=red_range(3) & hsv(:,:,2)<=red_range(4) ...
    & hsv(:,:,3)>=red_range(5) & hsv(:,:,3)<=red_range(6));

% Use the white mask to find the centroids and radii of the white circles
stats_white = regionprops('table', white_mask, 'Centroid', 'MajorAxisLength', 'MinorAxisLength');
centroids_white = stats_white.Centroid;
radii_white = (stats_white.MajorAxisLength + stats_white.MinorAxisLength)/4;

% Use the red mask to find the centroids and radii of the red circles
stats_red = regionprops('table', red_mask, 'Centroid', 'MajorAxisLength', 'MinorAxisLength');
centroids_red = stats_red.Centroid;
radii_red = (stats_red.MajorAxisLength + stats_red.MinorAxisLength)/4;

% Filter circles with radii outside of a certain range
min_radius = 15;
max_radius = 50;
idx_white = radii_white >= min_radius & radii_white <= max_radius;
centroids_white = centroids_white(idx_white, :);
radii_white = radii_white(idx_white);

idx_red = radii_red >= min_radius & radii_red <= max_radius;
centroids_red = centroids_red(idx_red, :);
radii_red = radii_red(idx_red);

% Draw circles around the white and red pieces
figure;
imshow(img);
hold on;
viscircles(centroids_white, radii_white, 'Color', 'b');
viscircles(centroids_red, radii_red, 'Color', 'b');

