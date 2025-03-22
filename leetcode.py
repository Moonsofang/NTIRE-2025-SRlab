from sklearn.tree import DecisionTreeClassifier
import numpy as np

def reverse(s):
    n = len(s)
    res = 0
    for i in range(n):
        status = 0
        len_zero = 0
        for j in range(i,n):
            if status==0:
                if s[j]=='1':
                    status = 2
                    res +=len_zero
                else:
                    status = 1
                    len_zero+=1
                    res += j-i+1-len_zero
            elif status == 1:
                if s[j]=='0':
                    len_zero+=1
                    res += j-i+1-len_zero
                else:
                    if len_zero*2==j-i+1:
                        status = 0
                        res+=len_zero
                    else:
                        res += j-i+1-len_zero
            else:
                if s[j]=='1':
                    res += len_zero
                else:
                    if len_zero==(j-i+1)//2:
                        status = 0
                        len_zero+=1
                        res+=len_zero
                    else:
                        len_zero+=1
                        res += len_zero
            print(i,j,status,res)
    return res
                        
                

# 示例输入


# 调用函数并输出结果
s = '0110'
print(reverse(s))