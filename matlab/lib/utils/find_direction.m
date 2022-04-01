function [ direction ] = find_direction(infield1, infield2)
%Find A->B or B->A
%   For output (direction), 0 = A->B, 1 = B->A, 2 = undetermined
    
    AB_list = [1 0 0 1; 1 0 1 1; 1 1 0 1];
    BA_list = [0 1 1 0; 0 1 1 1; 1 1 1 0];

    start_A = infield1(1); 
    end_A = infield1(end);
    start_B = infield2(1); 
    end_B = infield2(end);
    
    direction = 2;
    direction_mat = [start_A start_B end_A end_B];
    
    % Check AB
    for i=1:size(AB_list, 1)
        AB_vec = AB_list(i, :);
        
        result = isequal(AB_vec, direction_mat);
        
        if result == 1
            direction = 0;
            break;
        end
    end
    
    % Check BA
    for i=1:size(BA_list, 1)
        BA_vec = BA_list(i, :);
        
        result = isequal(BA_vec, direction_mat);
        
        if result == 1
            direction = 1;
            break;
        end
    end



end

