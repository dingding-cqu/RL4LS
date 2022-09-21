import os

def creat_data(dom_size, var1, var2, default_val):
    data = []
    for i in range(dom_size[var1]):
        data.append([default_val]*dom_size[var2])
    return data

def str2list(line):
    line = line.split()
    return [int(val) for val in line]

def str2list_head(line):
    line = line.split()
    line[0] = '-1'
    return [int(val) for val in line]


def parse(pth, scale):
    dom_size = dict()
    all_functions = []
    f = open(pth)
    line_head = f.readline()
    dom = str2list(f.readline())
    for i in range(len(dom)):
        dom_size[i] = dom[i]


    new_dom_size = dict(dom_size)
    while True:
        func_detail = f.readline()
        if len(func_detail) == 0:
            break
        func_detail = str2list(func_detail)
        if func_detail[0] == 2:
            data, var1, var2 = ext_2_ary_func(dom_size, f, func_detail)
            data = [[cell / scale for cell in row] for row in data ]
            all_functions.append((data, var1, var2))
        elif func_detail[0] == 1:
            data, var1 = ext_1_ary_func(dom_size, f, func_detail)
            data = [[cell / scale for cell in data]]
            var2 = len(new_dom_size)
            new_dom_size[var2] = 1
            all_functions.append((data, var2, var1))
        else:
            print('error in parse wcsp problems')
    return [(key,new_dom_size[key]) for key in new_dom_size.keys()], all_functions

def ext_1_ary_func(dom_size, f, func_detail):
    var1 = func_detail[1]
    default_val = func_detail[2]
    data = [default_val]*dom_size[var1]
    cnt = func_detail[3]
    while cnt > 0:
        cnt -= 1
        func_data = str2list(f.readline())
        val1 = func_data[0]
        data[val1] = func_data[1]
    return data, var1

def ext_2_ary_func(dom_size, f, func_detail):
    var1, var2 = func_detail[1], func_detail[2]
    default_val = func_detail[3]
    data = creat_data(dom_size, var1, var2, default_val)
    cnt = func_detail[4]
    while cnt > 0:
        cnt -= 1
        func_data = str2list(f.readline())
        val1, val2 = func_data[0], func_data[1]
        data[val1][val2] = func_data[2]
    return data, var1, var2

# if __name__ == '__main__':
#     # / tagsnp - 150
#     problems = []
#     path = '../real_world_problems'
#     for (dirpath, dirnames,files) in os.walk(path):
#         for filename in files:
#             if '.wcsp' in filename:
#                 filepath = os.path.join(dirpath, filename)
#                 problems.append(filepath)
#                 # dom_size, all_functions = parse_wcsp(filepath)
#                 # print(filepath, '\t', len(dom_size), '\t', len(all_functions))
