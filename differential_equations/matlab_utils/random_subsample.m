function subsampled_array = random_subsample(input_array, ratio_to_keep)
%remove removal_ratio of input_set's elements randomly
% Args:
%     input_array       -- [array] row or column vector
%     ratio_to_keep     -- [float] number between 0.0 and 1.0
% Returns:
%     subsampled_array  -- [array] subsampled input array 

n = length(input_array);
random_permutation = randperm(n);  % permutes integers 1:n
shuffled_array = input_array(random_permutation);
subsampled_array = shuffled_array(1:n*ratio_to_keep);

end