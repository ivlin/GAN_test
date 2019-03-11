import numpy as np
import matplotlib.pyplot as plt

train_sizes=[350,700,1400,2800,5600,11250,22500,55000]

def parse_csv(raw):
	accuracies=[]
	losses=[]
	raw_list=raw.split("\n")[1:]
	raw_list=[line.split(",")[:-1] for line in raw_list][:-1]
	print (raw_list)
	for item in raw_list:
		accuracies.append(float(item[0]))
		losses.append(float(item[1]))
	return accuracies, losses

for size in train_sizes:
	sup=open("./output/sup_train_size/%05d/testing_log"%size,"r")
	semisup=open("./output/semisup_train_size/%05d/testing_log"%size,"r")
	
	s_acc,s_loss=parse_csv(sup.read())
	ss_acc,ss_loss=parse_csv(semisup.read())

	plt.plot(np.linspace(0,len(s_acc),len(s_acc)),s_acc,label="supervised accuracy")
	plt.plot(np.linspace(0,len(ss_acc),len(ss_acc)),ss_acc,label="semisupervised accuracy")
	
	y_min,y_max=plt.ylim()

	axes = plt.gca()
	axes.set_ylim([ min(s_acc[1:]+ss_acc[1:]) ,1.0])
	
	plt.title("Training Progress")
	plt.xlabel("Iteration (10s)")
	plt.legend()
	plt.savefig("./output/train_size_comp/%05d_accuracy.png"%(size))
	plt.clf()

	axes.set_ylim(y_min,y_max)

	plt.plot(np.linspace(0,len(s_loss),len(s_loss)),s_loss,label="supervised loss")
	plt.plot(np.linspace(0,len(ss_loss),len(ss_loss)),ss_loss,label="semisupervised loss")
         
	plt.title("Training Progress")
	plt.xlabel("Iteration (10s)")
	plt.legend()
	plt.savefig("./output/train_size_comp/%05d_loss.png"%(size))
	plt.clf()