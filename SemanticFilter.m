function [F, E] = SemanticFilter(img, sigma_s, sigma_r, edge_model, num_iterations, joint_image)

    %% Setting up the arguments
    I = double(img);
    if ~exist('num_iterations', 'var')
        num_iterations = 3;
    end
    
    if exist('joint_image', 'var')
        J = double(joint_image);
    
        if (size(I,1) ~= size(J,1)) || (size(I,2) ~= size(J,2))
            error('Input and joint images must have equal width and height.');
        end
    else
        J = I;
    end
    
    [h, w, num_joint_channels] = size(J);



    
    %% Perform the filtering.
    N = num_iterations;
    F = I;
    
    sigma_H = sigma_s;
    
    for i = 0:num_iterations - 1

    % Apply Median filter
    % mfs   : median frame size
    mfs = 5;
    if mfs > 1
        
        M = medfiltimg(F, [mfs mfs]);
        mfs = mfs-2;
    else
        M=F;
    end

    % Estimate horizontal and vertical partial derivatives using finite
    % differences.
    dIcdx = diff(M, 1, 2);
    dIcdy = diff(M, 1, 1);
    
    dIdx = zeros(h,w);
    dIdy = zeros(h,w);
    
    % Compute the l1-norm distance of neighbor pixels.
    for c = 1:num_joint_channels
        dIdx(:,2:end) = dIdx(:,2:end) + abs( dIcdx(:,:,c) );
        dIdy(2:end,:) = dIdy(2:end,:) + abs( dIcdy(:,:,c) );
    end


        % detect edge probabilities and get map
        E = edgesDetect(F,edge_model);

        % Compute the derivatives of the horizontal and vertical domain transforms.
        dHdx = (1 + sigma_s/sigma_r * (E .* dIdx));
        dVdy = (1 + sigma_s/sigma_r * (E .* dIdy))';
    
        % Compute the sigma value for this iteration
        sigma_H_i = sigma_H * sqrt(3) * 2^(N - (i + 1)) / sqrt(4^N - 1);
    
        F = TransformedDomainRecursiveFilter_Horizontal(F, dHdx, sigma_H_i);
        F = image_transpose(F);
    
        F = TransformedDomainRecursiveFilter_Horizontal(F, dVdy, sigma_H_i);
        F = image_transpose(F);
        
    end
    
    F = cast(F, class(img));

end

%% Recursive filter.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function F = TransformedDomainRecursiveFilter_Horizontal(I, D, sigma)

    % Feedback coefficient 
    a = exp(-sqrt(2) / sigma);
    
    F = I;
    V = a.^D;
    
    [h, w, num_channels] = size(I);
    
    % Left -> Right filter.
    for i = 2:w
        for c = 1:num_channels
            F(:,i,c) = F(:,i,c) + V(:,i) .* ( F(:,i - 1,c) - F(:,i,c) );
        end
    end
    
    % Right -> Left filter.
    for i = w-1:-1:1
        for c = 1:num_channels
            F(:,i,c) = F(:,i,c) + V(:,i+1) .* ( F(:,i + 1,c) - F(:,i,c) );
        end
    end

end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function T = image_transpose(I)

    [h, w, num_channels] = size(I);
    
    T = zeros([w h num_channels], class(I));
    
    for c = 1:num_channels
        T(:,:,c) = I(:,:,c)';
    end
    
end

function M = medfiltimg(I, kernel)
    [h, w, num_channels] = size(I);

    M = zeros([h w num_channels], class(I));

    for c = 1:num_channels
        M(:,:,c) = medfilt2(I(:,:,c), kernel);
    end

end