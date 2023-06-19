function [x, y] = VehicleShape(x, y, theta, l, w)
    
    box = [x-l/2, y+w/2; x+l/2, y+w/2; x+l/2, y-w/2; x-l/2, y-w/2];
    box_matrix = box - repmat([x, y], size(box, 1), 1);
    theta = -theta;
    rota_matrix = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    new = box_matrix * rota_matrix + repmat([x, y], size(box, 1), 1);
    
    x = [new(1,1), new(2,1), new(3,1), new(4,1), new(1,1)];
    y = [new(1,2), new(2,2), new(3,2), new(4,2), new(1,2)];

end
