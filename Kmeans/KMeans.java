package HomeWork7;

import java.util.ArrayList;
import java.util.Random;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class KMeans {
	
	Instances centroids;
	int K;
	
	public KMeans(int K, Instances objects) {
		this.K = K;
		this.centroids = initEmptyCentroids(objects, K);
	}
	
	Instances initEmptyCentroids(Instances objects, int K){
		Instances emptyCentroids = new Instances(objects, K);
		for(int k = 0; k < K; k++){
			Instance centroid = new DenseInstance(objects.get(0));
			ArrayList<Attribute> attributes = new ArrayList<Attribute>(objects.get(0).numAttributes());
			for(int i = 0; i < objects.get(0).numAttributes(); i++){
				attributes.add(objects.get(0).attribute(i));
			}
						
			for(int i = 0; i < centroid.numAttributes(); i++){
				centroid.setValue(attributes.get(i), 0);
			}
			emptyCentroids.add(centroid);
		}
		
		return emptyCentroids;
	}
	
	
	/*
	 * Input: Instances object
	This method is building the KMeans object. It should initialize centroids 
	(by calling initializeCentroids) and run the K-Means algorithm 
	(which means to call findKMeansCentroids methods).
	 */
	void buildClusterModel(Instances objects){
		this.initializeCentroids(objects);
		this.findKMeansCentroids(objects);
		return;
	}
	
	/*
	 * Initialize the centroids by selecting k random instances 
	 * from the training set and setting the centroids to be those instances.
	 */
	void initializeCentroids(Instances objects){
		Random rng = new Random(); 
		ArrayList<Integer> randomIndexes = new ArrayList<Integer>();
		while (randomIndexes.size() < this.K) {
		    Integer next = rng.nextInt(objects.size()-1) + 1;
		    if(!randomIndexes.contains(next)){
		    	randomIndexes.add(next);
		    }
		}
				
		for(int k = 0; k < this.K; k++){
			for(int i = 0; i < objects.numAttributes(); i++){
				Attribute att = objects.attribute(i);
				this.centroids.get(k).setValue(att, objects.get(randomIndexes.get(k)).value(att));
			}
		}
	}
	
	/*
	 * Should find and store the centroids according to the KMeans algorithm. 
	 * Your stopping condition for when to stop iterating can be either when the centroids
	 * have not moved much from their previous location, the cost function did not change
	 * much, or you have reached a preset number of iterations. In this assignment we will
	 * only use the preset number option. A good preset number of iterations is 40. Use one or
	 * any combination of these methods to determine when to stop iterating.
	 * 
	 */
	void findKMeansCentroids(Instances objects) {
		int t = 0;
		int N = objects.size();
		int [] curAssigns = new int [N];
		
		while (t < 40){
			// assign each object to closest cluster
			for(int n = 0; n < N; n++){
				curAssigns[n] = findClosestCentroid(objects.get(n));
			}
			
			//re-calculate cluster centroids
			this.centroids = initEmptyCentroids(objects, this.K);
			
			for(int k = 0; k < this.K; k++){
				Instance cur_centroid = this.centroids.get(k);
				int count = 0;
				for(int n = 0; n < N; n++){
					if(curAssigns[n] == k){
						count += 1;
						this.addInstanceToInstance(cur_centroid, objects.get(n));
					}
				}
				this.divideInstanceByNumber(cur_centroid, count);
				
			}
			
			if(this.K == 5){
				System.out.println(this.calcAvgWSSSE(objects));
			}
			
			t += 1;		
		}
	}

	void addInstanceToInstance(Instance o1, Instance o2){
		for(int i = 0; i < o1.numAttributes(); i++){
			o1.setValue(o1.attribute(i), o1.value(o1.attribute(i)) + o2.value(o2.attribute(i)));
		}
		return;
	}
	
	void divideInstanceByNumber(Instance o1, int number){
		for(int i = 0; i < o1.numAttributes(); i++){
			o1.setValue(o1.attribute(i), 1.0 * o1.value(o1.attribute(i)) / number);
		}
		return;
	}
	
	double calcSquaredDistanceFromCentroid(Instance object, Instance centroid){
		double sum = 0.0;
		double temp = 0.0;
		for(int i = 0; i < object.numAttributes(); i++){
			temp = object.value(object.attribute(i)) - centroid.value(object.attribute(i));
			sum += temp * temp;
		}
		return sum;
	}
	

	/*
	 * a. Input: Instance
	 * b. Output: the index of the closest centroid to the input instance
	 */
	int findClosestCentroid(Instance object){
		int curIndex = 0;
		for(int k = 1; k < this.K; k++){
			double curDis = calcSquaredDistanceFromCentroid(object, this.centroids.get(k));
			double curClosestDis = calcSquaredDistanceFromCentroid(object, this.centroids.get(curIndex));
			if(curDis < curClosestDis){
				curIndex = k;
			}
		}
		return curIndex;
	}
		
	/*
	 * a. Input: Instances
	 * b. Output: should replace every instance in Instances by the centroid 
	 * to which it is assigned (closest centroid) and return the new Instances object.
	 */
	Instances quantize(Instances objects){
		
		Instances quantizedObjects = new Instances(objects, objects.size());
		for(int n = 0; n < objects.size(); n++){
			Instance object = objects.get(n);
			int kIndex = findClosestCentroid(object);
			
			
			Instance quantizedObject = new DenseInstance(object.numAttributes());
			
			ArrayList<Attribute> attributes = new ArrayList<Attribute>(object.numAttributes());
			for(int i = 0; i < object.numAttributes(); i++){
				attributes.add(object.attribute(i));
			}
						
			for(int i = 0; i < quantizedObject.numAttributes(); i++){
				quantizedObject.setValue(attributes.get(i), Math.floor(
						this.centroids.get(kIndex).value(attributes.get(i)))
				);
			}
			
			quantizedObjects.add(quantizedObject);
		}
		
		return quantizedObjects;
	}

	/*
	 * 	a. Input: Instances
	 * b. Output: should be the average within set sum of squared errors. That is it should 
	 * calculate the average squared distance of every instance from the centroid to which it 
	 * is assigned. This is Tr(Sc) from class, divided by the number of instances. Return the
	 * double value of the WSSSE.
	 */
	double calcAvgWSSSE(Instances objects){
		double error = 0.0;
		for(Instance object : objects){
			int centroidIndex = findClosestCentroid(object);
			double distance = calcSquaredDistanceFromCentroid(object, centroids.get(centroidIndex));
			error += distance;
		}
		error = error / objects.size();
		return error;
	}
			
}
