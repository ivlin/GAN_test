import numpy as np
import random as rand

def init_net():
    layers=[]
    layers.append([[rand.random(), rand.random()],
                   [rand.random(), rand.random()]])
    layers.append([[rand.random(), rand.random()]])
    return layers

#sigmoid
def activation(input_val, is_derivative):
    if is_derivative:
        return 1.0
    else:
        return input_val
    '''
    if not is_derivative:
        return 1.0/(1.0+np.exp(-1*input_val))
    return np.multiply(input_val,np.add(1.0,np.multiply(-1,input_val)))
    '''

def train(net, input_v, output_v, learn_rate):
    for i in xrange(10000):
        print "iteration", i
        out=[]
        cur_out=input_v[i%len(input_v)].T
        for layer in net:
            cur_out=activation(layer*cur_out,False)
            out.append(cur_out)
        print "    predicted: ", cur_out
        print "    actual: ", output_v[i%len(input_v)]

        l2_error=np.subtract(output_v[i%len(input_v)],cur_out)
        l2_gradient=activation(out[1],True)
        net[1]=np.add(net[1],learn_rate*np.multiply(l2_error,np.multiply(l2_gradient,out[0].T)))
        
        l1_gradient=activation(out[0],True)
        l1_error=np.multiply(l2_error,np.multiply(net[0],l1_gradient))
        net[0]=np.add(net[0],learn_rate*np.multiply(l1_error,np.multiply(l1_gradient,input_v[i%len(input_v)].T)))
        
if __name__ == "__main__":
    input_vals=np.matrix([[0.0,0.0],
                          [0.0,1.0],
                          [1.0,0.0],
                          [1.0,1.0]])
    output_vals=np.matrix([[0.0],[1.0],[1.0],[0.0]])
    net=init_net()
    train(net,input_vals,output_vals, 0.001)
