package HomeWork2;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;

class BasicRule {
	int attributeIndex;
	int attributeValue;
}

class Rule {
	List<BasicRule> basicRule;
	double returnValue;
}

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	double returnValue;
	Rule nodeRule = new Rule();

	public Node(Node[] children, Node parent, int attribute, double returnValue, Rule nodeRule) {
		this.children = children;
		this.parent = parent;
		this.attributeIndex = attribute;
		this.returnValue = returnValue;
		this.nodeRule = nodeRule;
	}

}

public class DecisionTree implements Classifier {
	private Node rootNode;

	public enum PruningMode {
		None, Chi, Rule
	};

	private PruningMode m_pruningMode;
	Instances validationSet;
	private List<Rule> rules = new ArrayList<Rule>();

	@Override
	public double classifyInstance(Instance data) {
		int attributeIndex;
		Node current = rootNode;
		int i;
		while (current.children != null) // not a leaf
		{
			attributeIndex = current.attributeIndex;
			i = (int) data.value(attributeIndex);
			if (current.children[i] == null) {
				return current.returnValue;
			}
			current = current.children[i];
		}
		return current.returnValue;
	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		rootNode = buildTree(arg0, null);
	}

	public Node buildTree(Instances data, Node parent) {
		Rule nodeRule = null;
		// if there is no instances
		if (data.numInstances() == 0) {
			double parentValue = parent.returnValue;
			Node leaf = new Node(null, parent, 0, parentValue, nodeRule);

			return leaf;
		}

		boolean equal = true;
		double classValue = data.instance(0).classValue();
		for (int i = 1; i < data.numInstances(); i++) {
			if (data.instance(i).classValue() != classValue) {
				equal = false;
				break;
			}
		}
		// if all instances have the same class value
		if (equal) {
			Node leaf = new Node(null, parent, 0, classValue, nodeRule);
			return leaf;
		}

		// If all instances have the same values
		boolean sameAttributeValues = true;
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			double firstInstanceValue = data.firstInstance().value(i);
			for (int j = 0; j < data.numInstances(); j++) {
				if (data.instance(j).value(i) != firstInstanceValue) {
					sameAttributeValues = false;
					break;
				}
			}
		}
		if (sameAttributeValues) {
			double majority = majorityOfInstances(data);
			Node leaf = new Node(null, parent, 0, majority, nodeRule);
			return leaf;

		}

		
		double bestInfoGain = Double.MIN_VALUE;
		int bestAttribute = -1;
		double currInfoGain;
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			currInfoGain = calcInfoGain(data, i);
			if (currInfoGain > bestInfoGain) {
				bestInfoGain = currInfoGain;
				bestAttribute = i;
			}
		}
		if (bestInfoGain <= 0 || bestAttribute == -1) {
			double majority = majorityOfInstances(data);
			Node leaf = new Node(null, parent, 0, majority, nodeRule);
			return leaf;
		}
		
		if (m_pruningMode==PruningMode.Chi) {
		double chi_square = calcChiSquare(data, bestAttribute);
		// need to check how to use the m_chiSquare
		if (chi_square <= 15.51) {
			double majority = majorityOfInstances(data);
			Node leaf = new Node(null, parent, 0, majority, nodeRule);
			return leaf;
		}
		}else {
			
		Instances[] splited = splitInstances(data, bestAttribute);
		
		Node[] children = new Node[splited.length];
		double majority = majorityOfInstances(data);

		for (int i = 0; i < children.length; i++) {
			if (splited[i].numInstances() > 0) {
				children[i] = buildTree(splited[i], rootNode);
			
			}

		}
		
		Node tree = new Node(children, parent, bestAttribute, majority, nodeRule);

		return tree;
		
		}//else if(m_pruningMode==PruningMode.Rule){
		//	return null;
		//}
		
		return null;
	}
	public double calcAvgError(Instances data) {
		double counter = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			Instance instance = data.instance(i);

			if (classifyInstance(instance) != instance.classValue())
				counter++;
		}
		return counter / data.numInstances();
	}

	public double calcInfoGain(Instances data, int attributeIndex) {
		double entropy = calcEntropy(data);
		Instances[] splited = splitInstances(data, attributeIndex);
		double weight = 0;

		for (int i = 0; i < splited.length; i++) {
			// Ignore empty split data sets
			if (splited[i].numInstances() == 0) continue;
			
			Instances current = splited[i];
			weight += (((double) current.numInstances() / data.numInstances()) * calcEntropy(current));
		}
		double infoGain = entropy - weight;
		return infoGain;
	}

	public static double calcEntropy(Instances data) {
		double class1value = data.instance(0).classValue();
		int numOfInstancesClass1 = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			if (data.instance(i).classValue() == class1value) {
				numOfInstancesClass1++;
			}
		}

		double p1 = (double) numOfInstancesClass1 / data.numInstances();
		double p2 = 1 - p1;
		double entropy = -p1 * log(p1) - p2 * log(p2);
		return entropy;
	}

	public double calcChiSquare(Instances data, int attributeIndex) {
		Instances[] splitedData = splitInstances(data, attributeIndex);
		double classValue = data.instance(0).classValue();
		int numOfclassInstances = 0;
		int counter;
		double chiSpuare = 0;
		double[] E0 = new double[splitedData.length];
		double[] E1 = new double[splitedData.length];
		double[] Pf = new double[splitedData.length];
		double[] Nf = new double[splitedData.length];
		double[] Df = new double[splitedData.length];
		double p1 = numOfclassInstances / data.numInstances();
		double p2 = 1 - p1;

		for (int i = 0; i < data.numInstances(); i++) {
			if (data.instance(i).classValue() == classValue)
				numOfclassInstances++;
		}
		
		for (int i = 0; i < Df.length; i++)
			Df[i] = splitedData[i].numInstances();
		

		// build array with Pf values.
		for (int i = 0; i < Pf.length; i++) {
			counter = 0;
			Instances currData = splitedData[i];
			for (int j = 0; j < currData.numInstances(); j++) {
				if (currData.instance(j).classValue() == classValue) {
					counter++;
				}
			}
			Pf[i] = counter;
		}

		for (int i = 0; i < Nf.length; i++) 
			Nf[i] = Df[i] - Pf[i];

		for (int i = 0; i < splitedData.length; i++) {
			E0[i] = Df[i] * p1;
			E1[i] = Df[i] * p2;
		}

		for (int i = 0; i < splitedData.length; i++) {
			if (E0[i] != 0 && E1[i] != 0)
				chiSpuare += ((Math.pow((E0[i] - Pf[i]), 2) / E0[i]) + (Math.pow(E1[i] - Nf[i], 2) / E1[i]));
		}
		return chiSpuare;
	}

	private static double log(double x) {
		if (x == 0) return 0;
			return Math.log(x) / Math.log(2);
	}

	public Instances[] splitInstances(Instances data, int attributeIndex) {
		int numValues = data.attribute(attributeIndex).numValues();
		String[] attributeValues = new String[numValues];
		Instances[] splited = new Instances[numValues];
		// create array and store all  the optional values for the attributes
		for (int i = 0; i < attributeValues.length; i++) {
			attributeValues[i] = data.attribute(attributeIndex).value(i);
		}
		// create Instances array. each Instances for one optional value
		for (int i = 0; i < splited.length; i++) 
			splited[i] = new Instances(data, 0);
		
		// split the input instances into the match Instances
		for (int i = 0; i < data.numInstances(); i++) {
			String currentValue = data.instance(i).stringValue(attributeIndex);
			for (int j = 0; j < attributeValues.length; j++) {
				if (currentValue.equals(attributeValues[j])) 
					splited[j].add(data.instance(i));
			}
		}
		return splited;
	}

	public double majorityOfInstances(Instances data) {
		double classValue = data.instance(0).classValue();
		int counter = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			if (data.instance(i).classValue() == classValue)	
				counter++;
		}
		double probability = counter / data.numInstances();
		
		 if (probability < 0.5) {
			int i = 1;
			while (data.instance(i).classValue() == classValue) {
				i++;
			}
			return data.instance(i).classValue();
		} else return classValue;
	}

	public void setPruningMode(PruningMode pruningMode) {
		m_pruningMode = pruningMode;
	}

	public void setValidation(Instances validation) {
		validationSet = validation;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}

}
