function string_nonnegative = string_array_max(string_array)
% utility function for ensuring nonnegative DE system components

[N,L] = size(string_array);
string_nonnegative = repmat([''],N,L+7);
for i = 1:N
    % remove all spaces from the string (should all be at the end)
    string_trimmed = string_array(i,:);
    string_trimmed(string_trimmed==' ') = '';
    total_spaces = L - size(string_trimmed,2);
    % interpret original value as nonnegative
    string_nonnegative(i,:) = sprintf('max(0,%s)%s', string_trimmed, blanks(total_spaces));
end

end
