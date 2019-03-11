import os

for filename in os.listdir("."):
    fn = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]

    while len(fn)<4 or fn[-4]>'9' or fn[-4]<'0':
        ind=-3
        while fn[ind]>'9' or fn[ind]<'0':
            ind+=1
        fn=fn[:ind]+'0'+fn[ind:]
    os.rename(filename,fn+ext)
