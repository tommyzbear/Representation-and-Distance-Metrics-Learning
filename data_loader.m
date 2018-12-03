% load data and feature vector in matlab
load('D:\EEE Year4\Representation-and-Distance-Metrics-Learning\PR_data\cuhk03_new_protocol_config_labeled.mat')
feature = jsondecode(fileread('D:\EEE Year4\Representation-and-Distance-Metrics-Learning\PR_data\feature_data.json'));

unsorted_rank_list = [];
for i = 1: length(query_idx)
   idx = query_idx(i);
   sample_rank_list = [];
   temp_cam_id = camId(idx - 1);
   temp_label = labels(idx - 1);
   for j = 1: length(gallery_idx)
       g_idx = gallery_idx(j);
       if(camId(g_idx - 1) ~= temp_cam_id && labels(g_idx) ~= temp_label)
           sample_rank_list = cat(1, sample_rank_list, g_idx);
       end
   end
   unsorted_rank_list = [unsorted_rank_list, sample_rank_list];
end

