function [tok_1, tok_2] = run_across_pair( pass_each, field1, field2 )
    
    xmesh1 = field1.pf.X;  % meshes from field 1 and 2 shall be same
    ymesh1 = field1.pf.Y;
    
    mask1 = field1.mask;
    mask2 = field2.mask;
    mask_union = (mask1+mask2) > 0;
    mask_intersect = (mask1+mask2) == 2;
    
    tok_1 = map_to_mask(pass_each.x, pass_each.y, xmesh1, ymesh1, mask1);
    tok_2 = map_to_mask(pass_each.x, pass_each.y, xmesh1, ymesh1, mask2);


end

