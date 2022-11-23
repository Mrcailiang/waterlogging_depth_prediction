def loadData(filename):
    data = []
    """
    读写模式：
    r ：只读 
    r+ : 读写 
    w ： 新建（会对原有文件进行覆盖） 
    a ： 追加 
    b ： 二进制文件只读文件
    """
    with open(filename,'r',encoding='utf8') as f:
        for line in f:
            #line.split()返回一行数据，强制类型转换为float型数据
            data.append([float(v) for v in line.split()])
    return data
