package HomeWork5;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.core.Instances;
import weka.core.Instance;
public class SVM {
	public SMO m_smo;

	public SVM() {
		this.m_smo = new SMO();
	}
	
	public void buildClassifier(Instances instances) throws Exception{
		m_smo.buildClassifier(instances);
	}
	void setKernel(Kernel k){
		m_smo.setKernel(k);
	}
	
	void setC(double C){
		m_smo.setC(C);
	}
	double getC(){
		return m_smo.getC();
	}
	//return [TP, FP, TN, FN]
	public int[] calcConfusion(Instances instances) throws Exception{
		int TP = 0, FP = 0, TN = 0, FN = 0;
		
		for(Instance element : instances){
			double trueClass = element.classValue();
			double predictedClass = m_smo.classifyInstance(element);

			if(trueClass == 1.0 && predictedClass == 1.0){
				TP += 1;
			} else if(trueClass == 1.0 && predictedClass == 0.0){
				FN += 1;
			} else if(trueClass == 0.0 && predictedClass == 0.0){
				TN += 1;
			} else if(trueClass == 0.0 && predictedClass == 1.0){
				FP += 1;
			}
		}
		
		int [] results = {TP, FP, TN, FN};
		return results;
	}
}
