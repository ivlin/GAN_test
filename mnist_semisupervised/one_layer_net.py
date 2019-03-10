import numpy as np
import random as rand

def init_net():
    layers=[]
    layers.append(np.subtract([[rand.random(), rand.random(), rand.random()],
                               [rand.random(), rand.random(), rand.random()],
                               [rand.random(), rand.random(), rand.random()],
                               [rand.random(), rand.random(), rand.random()]],1))
    layers.append(np.subtract([[rand.random(), rand.random(), rand.random(), rand.random()]],1))
    return layers

#sigmoid
def activation(input_val, is_derivative):
    if not is_derivative:
        return 1.0/(1.0+np.exp(-1*input_val))
    return np.multiply(input_val,np.add(1.0,np.multiply(-1,input_val)))

def train(net, input_v, output_v, learn_rate):
    for i in xrange(1000):
        #print("iteration", i)
        out=[]
        cur_out=input_v.T
        for layer in net:
            cur_out=activation(layer*cur_out,False)
            out.append(cur_out)
        #print("    predicted: ", cur_out)
        #print("    actual: ", output_v[i%len(input_v)])
        print("    err:", str(np.mean(np.abs(np.subtract(output_v,cur_out)))))

        l2_error=np.subtract(output_v,cur_out.T)
        l2_gradient=activation(out[1],True)
        net[1]=np.add(net[1],learn_rate*np.multiply(np.multiply(l2_error,l2_gradient),out[0].T))
        
        l1_gradient=activation(out[0],True)
        net[0]=np.add(net[0],learn_rate*np.multiply(np.multiply(np.multiply(l2_error,l2_gradient),net[1]),l1_gradient.T).T*input_v)
        
if __name__ == "__main__":
    input_vals=np.matrix([[0,0,1],
                          [0,1,1],
                          [1,0,1],
                          [1,1,1]])
    output_vals=np.matrix([[0.0],[1.0],[1.0],[0.0]])
    net=init_net()
    train(net,input_vals,output_vals, 0.1)
