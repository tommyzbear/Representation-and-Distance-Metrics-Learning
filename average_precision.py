def average_precision(labels_arr, correct_label):
    count = 0
    accumulated_precision = 0
    for i in range(len(labels_arr)):
        if labels_arr[i] == correct_label:
            count += 1
            accumulated_precision += count / (i + 1)
    if count != 0:
        return accumulated_precision / count
    else:
        return 0
