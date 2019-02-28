def get_name2(namelist):
    a = namelist
    length = len(a)
    if length == 1:
        return 0
    else:
        num = []
        sorted_num = []
        index = []
        names = []
        for j in a:
            num.append(int(j[-5]))
        sorted_num = sorted(num)

        for i in range(length-1):
            if (sorted_num[i+1]-sorted_num[i]) == 1:
                index.append(num.index(sorted_num[i]))
                index.append(num.index(sorted_num[i+1]))
        for k in index:
            names.append(a[k])
    return names


    #a = num.index(sorted_num[0])

if __name__ == '__main__':
    a = ['HELEN_1218567979_3_2.jpg', 'HELEN_1218567979_3_3.jpg', 'HELEN_1218567979_3_1.jpg', 'HELEN_1218567979_3_0.jpg']
    b = get_name2(a)
    print(b)
