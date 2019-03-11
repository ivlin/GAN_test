def draw_ascii(arr, filename="img.ascii", printOut=False):
    with open(filename,"w") as f:
        for i in xrange(len(arr)):
            if i%28==0:
                f.write('\n')
            if arr[i]>0:
                f.write("0")
            else:
                f.write(".")
    if printOut:
        with open(filename,"r") as f:
            print(f.read())
