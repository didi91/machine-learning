package HomeWork5;
/**
Didi Jungreisz
304993553
    &
Barak Gelman
204038756
**/
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Random;

import weka.core.Instances;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;

import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;
import weka.core.Instance;

public class MainHW5 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		DecimalFormat df = new DecimalFormat("0.00");
		
		Instances data = MainHW5.loadData("cancer.txt");

		// split data into train and test
		int trainSize = (int) Math.round(data.numInstances() * 80 / 100);
		int testSize = data.numInstances() - trainSize;
		Instances train = new Instances(data, 0, trainSize);
		Instances test = new Instances(data, trainSize, testSize);

		// true positive rate = TPR = (#TP)/(#P) = (#TP) / (#TP + #FN)
		// false positive rate	= FPR =	(#FP)/(#N) = (#FP) / (#FP + #TN)
		
		int [] results  = {0,0,0,0}; //[TP, FP, TN, FN]
		double TPR = 0.0, FPR = 0.0;
		
		double degree = 0;
		SVM classifier = new SVM();
		
		
		//=======polykernel======================
		double [] polyKerknelModelScores ={0,0,0}; 
		
		PolyKernel polyKernel = new PolyKernel();
		degree = 2;
		polyKernel.setExponent(degree);
		classifier.setKernel(polyKernel);
		classifier.buildClassifier(train);
		
		results = classifier.calcConfusion(test);
		TPR = 1.0 * results[0] / (results[0] + results[3]);
		FPR = 1.0 * results[1] / (results[1] + results[2]);
		polyKerknelModelScores[0] = TPR - FPR;
		System.out.println("For PolyKernel with degree " + degree + " the rates are:");
		System.out.println("TPR = " + df.format(TPR));
		System.out.println("FPR = " + df.format(FPR));
		System.out.println("\n");
		
		
		degree = 3;
		polyKernel.setExponent(degree);
		classifier.setKernel(polyKernel);
		classifier.buildClassifier(train);
		
		results = classifier.calcConfusion(test);
		TPR = 1.0 * results[0] / (results[0] + results[3]);
		FPR = 1.0 * results[1] / (results[1] + results[2]);
		polyKerknelModelScores[1] = TPR - FPR;
		System.out.println("For PolyKernel with degree " + degree + " the rates are:");
		System.out.println("TPR = " + df.format(TPR));
		System.out.println("FPR = " + df.format(FPR));
		System.out.println("\n");

		degree = 4;
		polyKernel.setExponent(degree);
		classifier.setKernel(polyKernel);
		classifier.buildClassifier(train);
		
		results = classifier.calcConfusion(test);
		TPR = 1.0 * results[0] / (results[0] + results[3]);
		FPR = 1.0 * results[1] / (results[1] + results[2]);
		polyKerknelModelScores[2] = TPR - FPR;
		System.out.println("For PolyKernel with degree " + degree + " the rates are:");
		System.out.println("TPR = " + df.format(TPR));
		System.out.println("FPR = " + df.format(FPR));
		System.out.println("\n");

		//==================================///
		double [] rbfKerknelModelScores = {0,0,0};
		
		RBFKernel rbfKernel = new RBFKernel();
		
		double gamma = 0.01;
		rbfKernel.setGamma(gamma);
		classifier.setKernel(rbfKernel);
		classifier.buildClassifier(train);
		
		results = classifier.calcConfusion(test);
		TPR = 1.0 * results[0] / (results[0] + results[3]);
		FPR = 1.0 * results[1] / (results[1] + results[2]);
		rbfKerknelModelScores[0] = TPR - FPR;
		System.out.println("For RBFKernel with gamma " + gamma + " the rates are:");
		System.out.println("TPR = " + df.format(TPR));
		System.out.println("FPR = " + df.format(FPR));
		System.out.println("\n");
		
		gamma = 0.1;
		rbfKernel.setGamma(gamma);
		classifier.setKernel(rbfKernel);
		classifier.buildClassifier(train);
		
		results = classifier.calcConfusion(test);
		TPR = 1.0 * results[0] / (results[0] + results[3]);
		FPR = 1.0 * results[1] / (results[1] + results[2]);
		rbfKerknelModelScores[1] = TPR - FPR;
		System.out.println("For RBFKernel with gamma " + gamma + " the rates are:");
		System.out.println("TPR = " + df.format(TPR));
		System.out.println("FPR = " + df.format(FPR));
		System.out.println("\n");
		
		gamma = 1;
		rbfKernel.setGamma(gamma);
		classifier.setKernel(rbfKernel);
		classifier.buildClassifier(train);
		
		results = classifier.calcConfusion(test);
		TPR = 1.0 * results[0] / (results[0] + results[3]);
		FPR = 1.0 * results[1] / (results[1] + results[2]);
		rbfKerknelModelScores[2] = TPR - FPR;
		System.out.println("For RBFKernel with gamma " + gamma + " the rates are:");
		System.out.println("TPR = " + df.format(TPR));
		System.out.println("FPR = " + df.format(FPR));
		System.out.println("\n");

		int maxPolyKernelIndex = 0;
		for(int i = 0; i < 3; i++){
			if(polyKerknelModelScores[maxPolyKernelIndex] < polyKerknelModelScores[i]){
				maxPolyKernelIndex = i;
			}
		}
		int maxRBFKernekIndex = 0;
		for(int i = 0; i < 3; i++){
			if(rbfKerknelModelScores[maxPolyKernelIndex] < rbfKerknelModelScores[i]){
				maxRBFKernekIndex = i;
			}
		}

		if(polyKerknelModelScores[maxPolyKernelIndex] > rbfKerknelModelScores[maxRBFKernekIndex]){
			System.out.print("The best kernel is Poly Kernel with  degree = ");
			String degree_str = "";
			if(maxPolyKernelIndex == 0) {
				degree_str = "2";
			} else if(maxPolyKernelIndex == 1){
				degree_str = "3";
			} else {
				degree_str = "4";
			}
			
			System.out.print(degree_str);
			System.out.println(". TPR - FPR = " + polyKerknelModelScores[maxPolyKernelIndex]);
		} else {
			System.out.print("The best kernel is RBFKernel with  gamma = ");
			String gamma_str = "";
			if(maxRBFKernekIndex == 0) {
				gamma_str = "0.01";
			} else if(maxRBFKernekIndex == 1){
				gamma_str = "0.1";
			} else {
				gamma_str = "1";
			}
			
			System.out.print(gamma_str);
			System.out.println(". TPR - FPR = " + polyKerknelModelScores[maxPolyKernelIndex]);
		}
		
		
		
		//==========find best slack value ================//
		degree = 3;
		polyKernel.setExponent(degree);
		classifier.setKernel(polyKernel);
		System.out.println("\n================\n");
		
		double [] C_values = {0.000033, 0.000066, 0.0001, 0.00033, 0.00066, 0.001,
				0.0033, 0.0066, 0.01, 0.033, 0.066, 0.1, 0.33, 0.66, 1, 3.33, 6.66, 10 };

		for(double C : C_values){
			classifier.setC(C);
			classifier.buildClassifier(train);
			
			results = classifier.calcConfusion(test);
			TPR = 1.0 * results[0] / (results[0] + results[3]);
			FPR = 1.0 * results[1] / (results[1] + results[2]);
			rbfKerknelModelScores[2] = TPR - FPR;
			System.out.println("For C =  " + C + " the rates are:");
			System.out.println("TPR = " + TPR);
			System.out.println("FPR = " + FPR);
			System.out.println("\n");
		}
		

	}
}
