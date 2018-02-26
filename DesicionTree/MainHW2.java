package HomeWork2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import HomeWork2.DecisionTree.PruningMode;
import weka.core.Instances;

public class MainHW2 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
	
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	

	public static void main(String[] args) throws Exception {
		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");
		DecisionTree tree = new DecisionTree();
		tree.buildClassifier(trainingCancer);
		double trainingError = tree.calcAvgError(trainingCancer);
		double testingError = tree.calcAvgError(testingCancer);
		tree.setPruningMode(PruningMode.None);
		System.out.println("Decision Tree with No pruning");
		System.out.println("The average train error of the decision tree is "+ trainingError);
		System.out.println("The average test error of the decision tree is "+ testingError);
		tree.setPruningMode(PruningMode.Chi);
		tree.buildClassifier(trainingCancer);
		trainingError = tree.calcAvgError(trainingCancer);
		testingError = tree.calcAvgError(testingCancer);
		System.out.println("Decision Tree with Chi pruning");
		System.out.println("The average train error of the decision tree with Chi pruning is "+ trainingError);
		System.out.println("The average test error of the decision tree with Chi pruning is "+ testingError);
	}
}
