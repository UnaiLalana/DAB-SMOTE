import numpy as np
from collections import OrderedDict

def dataReading(fileE):
	f = open(fileE, 'r')
	line = f.readline()
	att = OrderedDict()
	ds = dataset()
	ds.data = []
	ds.target = []
	ds.feature_names = []
	ds.target_names = []
	ds.categorical = []
	while line.find("@data") < 0:
		if line.find("@attribute") >= 0 and ((line.find("real") >= 0) or (line.find("integer") >= 0)):
			line = line.split()
			minVal = line[3].split("[")
			minVal = float(minVal[1].split(",")[0])
			maxVal = float(line[4].split("]")[0])
			attAux = {line[1]: [minVal, maxVal]}
			att.update(attAux)
			ds.categorical.append(False)
		elif line.find("@attribute") >= 0 and line.find("real") < 0:
			line = line.split()
			values = []
			l = line[2]
			l = l.split('{')
			values.append(l[1].split(',')[0])
			for l in line[3:-1]:
				values.append(l.split(',')[0])
			l = line[-1]
			l = l.split('}')
			values.append(l[0])
			attAux = {line[1]: values}
			att.update(attAux)
			ds.categorical.append(True)
		elif line.find("@output") >= 0 or line.find("@outputs") >= 0:
			class_name = line.split()
			class_name = class_name[1]
		line = f.readline()
	auxClasses = att.pop(class_name)
	ds.categorical = ds.categorical[:-1]
	classes = auxClasses[:]
	attAux = {class_name: classes}
	att.update(attAux)
	line = f.readline()
	exampleClasses = []
	examples = []
	examplesOriginal = []
	while line != "":
		line = line.replace(",", " ")
		l = line.split()
		values = l[0:len(l) - 1]
		val = []
		valOriginal = []
		for i, v in enumerate(values):
			if ds.categorical[i]:
				keyList = list(att)
				lista = att[keyList[i]]
				val.append(lista.index(v))
				valOriginal.append(v)
			else:
				val.append(float(v))
				valOriginal.append(float(v))
		examples.append(val)
		examplesOriginal.append(valOriginal)
		lista = att[class_name]
		exampleClasses.append(lista.index(l[len(l) - 1]))
		line = f.readline()
	examples = np.array(examples)
	f.close()
	ds.data = examples
	ds.target = np.array(exampleClasses)
	aux = list(att)
	ds.feature_names = aux[:-1]
	ds.target_names = att[class_name]
	ds.infoAtributos = att
	return ds

class dataset:
	data = []
	target = []
	feature_names = []
	target_names = []
	infoAtributos = {}
