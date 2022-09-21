def transpose(matrix):
    data = []
    # if matrix[0] is not list:
    #     for i in range(len(matrix)):
    #         data.append(matrix[i])
    # else:
    for col in range(len(matrix[0])):
        data.append([matrix[i][col] for i in range(len(matrix))])
    return data