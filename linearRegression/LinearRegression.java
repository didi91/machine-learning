
package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {

	private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;

	// the method which runs to train the linear regression predictor, i.e.
	// finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		trainingData = new Instances(trainingData);
		m_ClassIndex = trainingData.classIndex();
		// since class attribute is also an attribute we subtract 1
		m_truNumAttributes = trainingData.numAttributes() - 1;
		setAlpha(trainingData);
		m_coefficients = gradientDescent(trainingData);

	}

	private void setAlpha(Instances data) throws Exception {
		double[] tempTheta;
		double lError; // last error that was found
		double cError; // current error
		double sum;
		double factorial;
		double alpha = 0;
		double error = Double.MAX_VALUE;// previous alpha error
		for (int i = -17; i <= 2; i++) {
			m_coefficients = new double[m_truNumAttributes + 1];
			tempTheta = new double[m_truNumAttributes + 1];

			// sets the last error and the current error for each iteration of
			// alpha
			m_alpha = Math.pow(3, i);
			lError = 0;
			cError = Double.MAX_VALUE;
			// stops in 20000 iterations or if the difference between the last
			// error and the current in less than 0.003
			for (int k = 0; k <= 20000 && Math.abs(lError - cError) > 0.003; k++) {
				for (int l = 0; l < m_truNumAttributes + 1; l++) {
					sum = 0; // set sum to zero for each new column
					for (int m = 0; m < data.numInstances(); m++) {
						// the factorial for each argument
						factorial = (l == 0) ? 1 : data.instance(m).value(l - 1);
						// sum all the partial derivative of each raw
						sum += (regressionPrediction(data.instance(m)) - data.instance(m).value(m_ClassIndex))
								* factorial;
					}
					// computes gradient descent formula for regression
					tempTheta[l] = (m_coefficients[l] - (m_alpha / data.numInstances()) * sum);
				}
				// copies data from tempTheta to m_coefficients
				System.arraycopy(tempTheta, 0, m_coefficients, 0, m_coefficients.length);
				lError = cError;
				cError = calculateSE(data);
			}
			if (cError > error) {
				m_alpha = alpha;
				break;
			} else {
				error = cError;
				alpha = m_alpha;
			}
		}
	}

	/**
	 * An implementation of the gradient descent algorithm which should return
	 * the weights of a linear regression predictor which minimizes the average
	 * squared error.
	 * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData) throws Exception {
		double lError = 0; // last error that was found
		double cError = calculateSE(trainingData); // current error
		double sum;
		double factorial;
		double[] tempTheta = new double[m_truNumAttributes + 1];
		m_coefficients = new double[m_truNumAttributes + 1];
		// Initialize all coefficients with zero
		for (int i = 0; i < m_coefficients.length; i++)
			m_coefficients[i] = 0;
		for (int k = 0;; k++) {
			for (int l = 0; l < m_truNumAttributes + 1; l++) {
				sum = 0; // set sum to zero for each new column
				for (int m = 0; m < trainingData.numInstances(); m++) {
					// the factorial for each argument
					factorial = (l == 0) ? 1 : trainingData.instance(m).value(l - 1);
					// sum all the partial derivative of each raw
					sum += (regressionPrediction(trainingData.instance(m))
							- trainingData.instance(m).value(m_ClassIndex)) * factorial;
				}
				// computes gradient descent formula for regression
				tempTheta[l] = (m_coefficients[l] - (m_alpha / trainingData.numInstances()) * sum);
			}
			// copies data from tempTheta to m_coefficients
			System.arraycopy(tempTheta, 0, m_coefficients, 0, m_coefficients.length);
			if (k % 100 == 0) {
				if (Math.abs(lError - cError) < 0.003)
					break;
				lError = cError;
			}
			cError = calculateSE(trainingData);
		}
		for (int i = 0; i < m_coefficients.length; i++) {
			System.out.println(m_coefficients[i]);

		}
		return m_coefficients;
	}

	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
	 *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		double prediction = m_coefficients[0];
		int i = 0;
		while (i < m_truNumAttributes) {
			prediction += m_coefficients[i + 1] * instance.value(i);
			i++;
		}
		return prediction;
	}

	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
	 *
	 * @param testData
	 * @return
	 * @throws Exception
	 */
	public double calculateSE(Instances data) throws Exception {
		double sqrError = 0;
		int i = 0;
		while (i < data.numInstances()) {
			sqrError += Math.pow(regressionPrediction(data.instance(i)) - data.instance(i).value(m_ClassIndex), 2);
			i++;
		}
		return sqrError / data.numInstances();
	}

	@Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
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
