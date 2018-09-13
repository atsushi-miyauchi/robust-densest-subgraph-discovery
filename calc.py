#coding:utf-8

import numpy
import matplotlib.pyplot as plt
import math
import random
import heapq
from time import time
import os
import sys
from gurobipy import *



class Graph:

	def __init__(self):
		self.nodes = []
		self.adjlist = {}
	
	def add_node(self, v):
		self.nodes.append(v)
		self.adjlist[v] = {}

	def add_edge(self, u, v, w=1):
		self.adjlist[u][v] = w
		self.adjlist[v][u] = w
	
	def edges(self):
		E = []
		for u in self.nodes:
			for v in self.adjlist[u]:
				if u < v: E.append((u,v))
		return E

	def w_edges(self):
		E = []
		for u in self.nodes:
			for v in self.adjlist[u]: 
				if u < v: E.append((u,v,self.adjlist[u][v]))
		return E

	def degree(self, v):
		return len(self.adjlist[v])
	
	def w_degree(self, v):
		deg = 0
		for u in self.adjlist[v]: deg += self.adjlist[v][u]
		return deg

	def max_degree(self):
		return max([self.degree(v) for v in self.nodes])
	
	def w_max_degree(self):
		return max([self.w_degree(v) for v in self.nodes])

	def sum_of_weight(self):
		sum_of_weight = 0
		for (u,v,w) in self.w_edges():
			sum_of_weight += w
		return sum_of_weight

	def average_degree(self):
		return 2.*self.sum_of_weight() / len(self.nodes)

	def density(self, subset):
		num_of_nodes = 0
		sum_of_weight = 0
		for u in self.nodes:
			if subset[u] == 1:
				num_of_nodes += 1
				for v in self.adjlist[u]: 
					if subset[v] == 1 and u < v: 
						sum_of_weight += self.adjlist[u][v]
					#if subset[v] == 1: sum_of_weight += self.adjlist[u][v]
		return sum_of_weight / float(num_of_nodes)
		
	def Greedy_peeling(self):
		# O((m+n)logn)-time greedy peeling for the densest subgraph problem
		t1 = time()
		n = len(self.nodes) # # of current nodes
		m = self.sum_of_weight() # # of current edges
		n_max = max(self.nodes)
		deg_list = [0]*(n_max+1) # deg_list[v] = current degree of node v
		for v in self.nodes: deg_list[v] = self.w_degree(v)
		flag = [0]*(n_max+1)
		for v in self.nodes: flag[v] = 1 # means v still exists 
		heap = []
		for v in self.nodes: heap.append((deg_list[v],v))
		heapq.heapify(heap)
		max_val = m / float(n) 
		max_val_n = n
		max_val_m = m
		while n > 1:
			min_deg, v_min = heapq.heappop(heap)
			if flag[v_min] == 1:
				n -= 1
				m -= min_deg
				val = m / float(n)
				flag[v_min] = 0
				if val > max_val:
					max_val = val
					max_val_n = n
					max_val_m = m
				for v in self.adjlist[v_min]:
					if flag[v] == 1:
						deg_list[v] -= 1
						heapq.heappush(heap, (deg_list[v],v))
		t2 = time()
		print 'f(S) =', round(max_val, 5)
		print 'w(S) =', round(max_val_m, 5)
		print '|S| =', max_val_n 
		print "time(s):", t2 - t1

	def LP_exact(self):
		# Charikar's LP-based exact algorithm for the densest subgraph problem
		#t1 = time()
		n = len(self.nodes)
		m = self.sum_of_weight()
		model = Model()
		var_list = {}
		for v in self.nodes: var_list[v] = model.addVar(obj=0, vtype=GRB.CONTINUOUS, lb=0)
		for u, v in self.edges():
			var_list[u, v] = model.addVar(obj=self.adjlist[u][v], vtype=GRB.CONTINUOUS, lb=0)
		model.update()
		for u, v in self.edges():
			model.addConstr(LinExpr([1,-1], [var_list[u,v], var_list[u]]), GRB.LESS_EQUAL, 0)
			model.addConstr(LinExpr([1,-1], [var_list[u,v], var_list[v]]), GRB.LESS_EQUAL, 0)
		model.addConstr(LinExpr([1]*n, [var_list[v] for v in self.nodes]), GRB.EQUAL, 1)
		model.ModelSense = -1
		model.setParam("Method", 1) 
		#model.setParam("OutputFlag",0)
		model.optimize()
		if model.status == GRB.OPTIMAL: 
			opt_sol =  model.getAttr("X", var_list)
			r_list = []
			for v in self.nodes:
				if opt_sol[v] not in r_list: r_list.append(opt_sol[v])
			n_max = max(self.nodes)
			max_density = -1
			for r in r_list:
				subset = [0]*(n_max+1)
				for v in self.nodes: 
					if opt_sol[v] >= r: subset[v] = 1
				density = self.density(subset)
				if density > max_density:
					best_subset = subset
					max_density = density
		#t2 = time()
		#print 'f(S) =', round(max_density, 5)
		#print 'w(S) =', round(max_density*sum(best_subset), 5)
		#print '|S| =', sum(best_subset)
		#print "time(s):", t2 - t1
		return best_subset

	def LP_exact_fast(self):
		# Charikar's LP-based exact algorithm with Balalau et al.'s preprocessing
		# BEGIN greedy peeling
		#t1 = time()
		n = len(self.nodes) # # of current nodes
		m = self.sum_of_weight() # # of current edges
		n_max = max(self.nodes)
		deg_list = [0]*(n_max+1) # deg_list[v] = current degree of node v
		for v in self.nodes: deg_list[v] = self.w_degree(v)
		flag = [0]*(n_max+1)
		for v in self.nodes: flag[v] = 1 # means v still exists 
		heap = []
		for v in self.nodes: heap.append((deg_list[v],v))
		heapq.heapify(heap)
		max_val = m / float(n) 
		max_val_n = n
		max_val_m = m
		v_min_sequence = []
		counter = 0
		max_counter = 0
		while n > 1:
			min_deg, v_min = heapq.heappop(heap)
			if flag[v_min] == 1:
				n -= 1
				m -= min_deg
				val = m / float(n)
				flag[v_min] = 0
				v_min_sequence.append(v_min)
				counter += 1
				if val > max_val:
					max_val = val
					max_val_n = n
					max_val_m = m
					max_counter = counter
				for v in self.adjlist[v_min]:
					if flag[v] == 1:
						deg_list[v] -= self.adjlist[v][v_min]
						#deg_list[v] -= 1
						heapq.heappush(heap, (deg_list[v],v))
		peeling_solution = [0]*(n_max+1)
		for v in self.nodes: peeling_solution[v] = 1
		for v in v_min_sequence[:max_counter]: peeling_solution[v] = 0 # approximate solution by peeling
		ALG = self.density(peeling_solution)
		V = [0]*(n_max+1)
		for v in self.nodes: 
			if self.w_degree(v) >= ALG: V[v] = 1
		# END greedy peeling
		model = Model()
		var_list = {}
		counter = 0
		for v in self.nodes: 
			if V[v] == 1: 
				var_list[v] = model.addVar(obj=0, vtype=GRB.CONTINUOUS, lb=0)
				counter += 1
		for u, v in self.edges():
			if V[u] == 1 and V[v] == 1: 
				var_list[u, v] = model.addVar(obj=self.adjlist[u][v], vtype=GRB.CONTINUOUS, lb=0)
		model.update()
		for u, v in self.edges():
			if V[u] == 1 and V[v] == 1:
				model.addConstr(LinExpr([1,-1], [var_list[u,v], var_list[u]]), GRB.LESS_EQUAL, 0)
				model.addConstr(LinExpr([1,-1], [var_list[u,v], var_list[v]]), GRB.LESS_EQUAL, 0)
		model.addConstr(LinExpr([1]*sum(V), [var_list[v] for v in self.nodes if V[v] == 1]), GRB.EQUAL, 1)
		model.setParam("OutputFlag",0)
		model.ModelSense = -1
		model.setParam("Method", 1) 
		model.optimize()
		if model.status == GRB.OPTIMAL: 
			opt_sol =  model.getAttr("X", var_list)
			r_list = []
			for v in self.nodes:
				if V[v] == 1:
					if opt_sol[v] not in r_list: r_list.append(opt_sol[v])
			n_max = max(self.nodes)
			max_density = -1
			for r in r_list:
				subset = [0]*(n_max+1)
				for v in self.nodes: 
					if V[v] == 1:
						if opt_sol[v] >= r: subset[v] = 1
				density = self.density(subset)
				if density > max_density:
					best_subset = subset
					max_density = density
		#t2 = time()
		#print 'f(S) =', round(max_density, 5)
		#print 'w(S) =', round(max_density*sum(best_subset), 5)
		#print '|S| =', sum(best_subset)
		#print "time(s):", t2 - t1
		return best_subset



	def read(self, f_name):
		# read a graph
		f = open(f_name, 'r')
		lines = f.readlines()
		n_max = 0
		for line in lines:
			new_line = line.strip().split()
			for i in new_line:
				if int(i) > n_max: n_max = int(i)
		n_flag = [0]*(n_max+1)
		for line in lines:
			new_line = line.strip().split()
			for i in new_line: n_flag[int(i)] = 1
		for index, val in enumerate(n_flag): 
			if val == 1: self.add_node(index)
		for line in lines:
			new_line = line.strip().split()
			self.add_edge(int(new_line[0]), int(new_line[1]))
		f.flush()
		f.close()

	def read_loop_multiedge_proof(self, f_name):
		# read a graph ignoring loops and multiple edges
		f = open(f_name, 'r')
		lines = f.readlines()
		n_max = 0
		for line in lines:
			new_line = line.strip().split()
			for i in new_line:
				if int(i) > n_max: n_max = int(i)
		n_flag = [0]*(n_max+1)
		for line in lines:
			new_line = line.strip().split()
			for i in new_line: n_flag[int(i)] = 1
		for index, val in enumerate(n_flag): 
			if val == 1: self.add_node(index)
		for line in lines:
			new_line = line.strip().split()
			if int(new_line[0]) != int(new_line[1]) and int(new_line[1]) not in self.adjlist[int(new_line[0])]:
				self.add_edge(int(new_line[0]), int(new_line[1]))
		f.flush()
		f.close()

	def read_snap(self, f_name):
		# read a graph in SNAP
		f = open(f_name, 'r')
		lines = f.readlines()
		n_max = 0
		for line in lines:
			new_line = line.strip().split("\t")
			for i in new_line:
				if int(i) > n_max: n_max = int(i)
		n_flag = [0]*(n_max+1)
		for line in lines:
			new_line = line.strip().split("\t")
			for i in new_line: n_flag[int(i)] = 1
		for index, val in enumerate(n_flag): 
			if val == 1: self.add_node(index)
		for line in lines:
			new_line = line.strip().split('\t')
			self.add_edge(int(new_line[0]), int(new_line[1]))
		f.flush()
		f.close()

	def write(self, f_name):
		# write a graph
		f = open(f_name, 'w')
		for e in self.edges():
			f.write(str(e[0])+' '+str(e[1])+'\n')
		f.flush()
		f.close()





class Graph_interval:

	def __init__(self):
		self.nodes = []
		self.adjlist = {}
	
	def add_node(self, v):
		self.nodes.append(v)
		self.adjlist[v] = {}

	def add_edge(self, u, v, l=1, r=1, w_true=1):
		self.adjlist[u][v] = (l,r,w_true)
		self.adjlist[v][u] = (l,r,w_true)
	
	def edges(self):
		E = []
		for u in self.nodes:
			for v in self.adjlist[u]:
				if u < v: E.append((u,v))
		return E

	def w_edges(self):
		E = []
		for u in self.nodes:
			for v in self.adjlist[u]: 
				if u < v: E.append((u,v,self.adjlist[u][v]))
		return E

	def r_degree(self, v):
		r_degree = 0
		for u in self.adjlist[v]:
			r_degree += self.adjlist[v][u][1]
		return r_degree
			


	def Random(self):
		# "Random" in the paper
		# BEGIN algorithm
		t1 = time()
		G_random = Graph()
		for v in self.nodes: G_random.add_node(v)
		for (u,v,(l_e,r_e,w_true_e)) in self.w_edges(): 
			G_random.add_edge(u,v,numpy.random.uniform(l_e,r_e))
		subset_random = G_random.LP_exact_fast()
		t2 = time()
		# END algorithm
		G_true = Graph()
		for v in self.nodes: G_true.add_node(v)
		for (u,v,(l_e,r_e,w_true_e)) in self.w_edges(): G_true.add_edge(u,v,w_true_e)
		opt_subset = G_true.LP_exact_fast()
		robust_ratio_at_w_true = G_true.density(subset_random)/G_true.density(opt_subset)
		return robust_ratio_at_w_true, t2-t1

	def Proposed_without_sampling(self):
		# "Algorithm 1" in the paper
		# BEGIN algorithm
		t1 = time()
		G_w_l = Graph()
		for v in self.nodes: G_w_l.add_node(v)
		for (u,v,(l_e,r_e,w_true_e)) in self.w_edges(): G_w_l.add_edge(u,v,l_e)
		subset_proposed_nonsampling = G_w_l.LP_exact_fast()
		t2 = time()
		# END algorithm
		G_true = Graph()
		for v in self.nodes: G_true.add_node(v)
		for (u,v,(l_e,r_e,w_true_e)) in self.w_edges(): G_true.add_edge(u,v,w_true_e)
		opt_subset = G_true.LP_exact_fast()
		robust_ratio_at_w_true = G_true.density(subset_proposed_nonsampling)/G_true.density(opt_subset)
		return robust_ratio_at_w_true, t2-t1

	def Proposed_with_sampling(self, eps=0.9, gamma=0.9):
		# "Algorithm 2" in the paper
		# BEGIN algorithm
		t1 = time()
		n = len(self.nodes)
		m = len(self.edges())
		W_true = {}
		#for (u,v,(l_e,r_e,w_true_e)) in self.w_edges(): W_true[u,v] = w_true_e
		for (u,v) in self.edges(): W_true[u,v] = self.adjlist[u][v][2]
		#d_avg = 2.*m / n
		W_out = {}
		t_e_list = []
		G_w_l = Graph()
		for v in self.nodes: G_w_l.add_node(v)
		for (u,v,(l_e,r_e,w_true_e)) in self.w_edges(): G_w_l.add_edge(u,v,l_e)
		densest_subgraph_on_G_w_l = G_w_l.LP_exact_fast()
		LB = G_w_l.density(densest_subgraph_on_G_w_l)
		for (u,v,(l_e,r_e,w_true_e)) in self.w_edges():
			if l_e == r_e: W_out[u,v] = (l_e,r_e)
			else: 
				t_e = int(math.ceil(m*((r_e-l_e)**2.)*numpy.log(2.*m/gamma) / ((eps**2.)*(LB**2.))))
				t_e_list.append(t_e)
				sampling_delta = min(r_e-W_true[u,v],W_true[u,v]-l_e)
				left = W_true[u,v] - sampling_delta
				right = W_true[u,v] + sampling_delta
				p_e = 0
				for i in range(t_e): p_e += numpy.random.uniform(left,right)
				p_e /= t_e
				delta = eps*LB / math.sqrt(2*m)
				W_out[u,v] = (max(l_e,p_e-delta), min(r_e,p_e+delta))
		G_w_out_l = Graph()
		for v in self.nodes: G_w_out_l.add_node(v)
		for (u,v) in self.edges(): G_w_out_l.add_edge(u,v,W_out[u,v][0])
		densest_subgraph_on_G_w_out_l = G_w_out_l.LP_exact_fast()
		t2 = time()
		# END algorithm
		success = 1 # flag equal to 1 if w_true in W_out 0 otherwise
		for (u,v) in self.edges():
			if W_true[u,v] < W_out[u,v][0] or W_out[u,v][1] < W_true[u,v]: success = 0
		G_true = Graph()
		for v in self.nodes: G_true.add_node(v)
		for (u,v) in self.edges(): G_true.add_edge(u,v,W_true[u,v])
		opt_subset = G_true.LP_exact_fast()
		robust_ratio_at_w_true = G_true.density(densest_subgraph_on_G_w_out_l)/G_true.density(opt_subset)
		avg_num_sampling = sum(t_e_list)/float(m)
		return robust_ratio_at_w_true, avg_num_sampling, t2-t1, success, sum(t_e_list)

	def Proposed_with_sampling_fast(self, eps=0.9, gamma=0.9):
		# "Algorithm 2" in the paper with preprocessing
		# BEGIN algorithm
		t1 = time()
		n = len(self.nodes)
		m = len(self.edges())
		W_true = {}
		for (u,v,(l_e,r_e,w_true_e)) in self.w_edges(): W_true[u,v] = w_true_e
		W_out = {}
		t_e_list = []
		# BEGIN preprocessing
		G_w_l = Graph()
		for v in self.nodes: G_w_l.add_node(v)
		for (u,v,(l_e,r_e,w_true_e)) in self.w_edges(): G_w_l.add_edge(u,v,l_e)
		densest_subgraph_on_G_w_l = G_w_l.LP_exact_fast()
		LB = G_w_l.density(densest_subgraph_on_G_w_l)
		n_max = max(self.nodes)
		V = [0]*(n_max+1)
		for v in self.nodes: 
			if self.r_degree(v) >= LB: V[v] = 1
		new_m = 0
		for (u,v) in self.edges():
			if V[u] == 1 and V[v] == 1: new_m += 1
		#print m, new_m
		# END preprocessing
		for (u,v,(l_e,r_e,w_true_e)) in self.w_edges():
			if V[u] == 1 and V[v] == 1:
				if l_e == r_e: W_out[u,v] = (l_e,r_e)
				else: 
					t_e = int(math.ceil(new_m*((r_e-l_e)**2.)*numpy.log(2.*new_m/gamma) / ((eps**2.)*(LB**2.))))
					t_e_list.append(t_e)
					sampling_delta = min(r_e-W_true[u,v],W_true[u,v]-l_e)
					left = W_true[u,v] - sampling_delta
					right = W_true[u,v] + sampling_delta
					p_e = 0
					for i in range(t_e): p_e += numpy.random.uniform(left,right)
					p_e /= t_e
					delta = eps*LB / math.sqrt(2*new_m)
					W_out[u,v] = (max(l_e,p_e-delta), min(r_e,p_e+delta))
			else: W_out[u,v] = (l_e,r_e)
		G_w_out_l = Graph()
		for v in self.nodes: 
			if V[v] == 1: G_w_out_l.add_node(v)
		for (u,v) in self.edges(): 
			if V[u] == 1 and V[v] == 1: G_w_out_l.add_edge(u,v,W_out[u,v][0])
		densest_subgraph_on_G_w_out_l = G_w_out_l.LP_exact_fast()
		t2 = time()
		# END algorithm
		success = 1 # flag equal to 1 if w_true in W_out and 0 otherwise
		for (u,v) in self.edges():
			if V[u] == 1 and V[v] == 1:
				if W_true[u,v] < W_out[u,v][0] or W_out[u,v][1] < W_true[u,v]: success = 0
		G_true = Graph()
		for v in self.nodes: 
			if V[v] == 1: G_true.add_node(v)
		for (u,v) in self.edges(): 
			if V[u] == 1 and V[v] == 1: G_true.add_edge(u,v,W_true[u,v])
		opt_subset = G_true.LP_exact_fast()
		robust_ratio_at_w_true = G_true.density(densest_subgraph_on_G_w_out_l)/G_true.density(opt_subset)
		avg_num_sampling = sum(t_e_list)/float(new_m)
		return robust_ratio_at_w_true, avg_num_sampling, t2-t1, success, sum(t_e_list)





	def planted_dense_subgraph(self,n_c=50,alpha=0.0):
		# Generate a graph with an edge-weight space and a true edge-weight vector under the planted uncertain dense subgraph model
		n = 500
		for i in range(n): self.add_node(i)
		for u in self.nodes:
			for v in self.nodes:
				if u < v:
					if numpy.random.uniform() <= 0.01:
						if v < n_c:
							l = numpy.random.uniform(0.1+alpha,1.0)
							r = 1.0
							self.add_edge(u,v,l,r,numpy.random.uniform(max(l,0.9),r))
						else:
							l = 0.1
							r = numpy.random.uniform(0.1,1.0-alpha)
							self.add_edge(u,v,l,r,numpy.random.uniform(l,min(r,0.2)))



	def read(self, f_name):
		# read a graph with an edge-weight space and a true edge-weight vector
		f = open(f_name, 'r')
		lines = f.readlines()
		n_max = 0
		for line in lines:
			new_line = line.strip().split()
			for i in new_line[:2]:
				if int(i) > n_max: n_max = int(i)
		n_flag = [0]*(n_max+1)
		for line in lines:
			new_line = line.strip().split()
			for i in new_line[:2]: n_flag[int(i)] = 1
		for index, val in enumerate(n_flag): 
			if val == 1: self.add_node(index)
		for line in lines:
			new_line = line.strip().split()
			self.add_edge(int(new_line[0]), int(new_line[1]), float(new_line[2]), float(new_line[3]), float(new_line[4]))
		f.flush()
		f.close()
	
	def write(self, f_name):
		# write a graph with an edge-weight space and a true edge-weight vector
		f = open("./instance_interval/"+f_name, 'w')
		for (u,v,(l,r,w_true)) in self.w_edges():
			f.write(str(u)+" "+str(v)+" "+str(l)+" "+str(r)+" "+str(w_true)+"\n")
		f.flush()
		f.close()



def real_to_interval_knockout(instance):
	# generate a graph with an edge-weight space and a true edge-weight vector 
	# from a real-world graph under the knockout densest subgraph model
	G = Graph()
	G.read("./instance/"+instance)
	random.seed(0)
	f = open("./instance_interval_knockout/"+instance, 'w')
	opt_subset = G.LP_exact_fast()
	for e in G.edges():
		if opt_subset[e[0]] == 1 and opt_subset[e[1]] == 1:
			l = 0.1
			r = numpy.random.uniform(0.1,0.9)
			w_true = numpy.random.uniform(0.1,min(r,0.11))
		else: 
			l = numpy.random.uniform(0.2,1.0)
			r = 1.0
			w_true = numpy.random.uniform(max(l,0.99),1.0)
		f.write(str(e[0])+' '+str(e[1])+' '+str(l)+' '+str(r)+' '+str(w_true)+'\n')
	f.flush()
	f.close()
			
def real_to_interval_knockout_all():
	instances = os.listdir("./instance")
	for instance in instances:
		real_to_interval_knockout(instance)

