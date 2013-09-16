/*
*  kmeans.c 
* openMP version  - Kmeans parallelization inside the algorithm
* Asimina Vouronikoy - Eleni Zisiou
* compile : gcc -Wall kmeans.c -o kmeans -lm -fopenmp
* run :  ./kmeans
* 
*/

#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <time.h>

#define MAX_THREADS 1

void print_data(double **data_points,int num_points,int dimension){
	int i,j;

	for(i=0;i<num_points;i++){
		printf("[");
		for(j=0;j<dimension;j++)
			printf(" %.3lf",data_points[i][j]);
		printf(" ]\n");
	}

}

double calc_distance(double *data_points,double *c,int dimension){
	double distance =0,t;
	int j;
	for (j=0; j<dimension; j++){
		t = data_points[j] - c[j];
		distance += t*t;
	}
	return distance;
}

double calc_distance_et(double *data_points,double *c,int dimension,double min){
	double distance =0,t;
	int j;
	for (j=0; j<dimension; j++){
		t = data_points[j] - c[j];
		distance += t*t;
		if(distance>=min) break;
	}
	return distance;
}
double norm(double *vector,int dimension){
	double sum=0;
	int i;

	for(i=0; i<dimension; sum+=vector[i]*vector[i],i++);
	return sqrt(sum);
}

int check_match(double *data_points,double *c,int dimension){
	int j;

	for (j=0; j<dimension; j++){
		if(data_points[j] != c[j]) break;
	}

	if(j==dimension) return 1;
	else return 0;

}

double  **kkz_initialization(double **data_points,int num_points,int dimension,int num_of_clusters){
	double *normVector,*distance;
	double max=DBL_MIN, max_distance = DBL_MIN , min_distance = DBL_MAX,dist;
	int i,k,p,max_index,num_of_centroids=0,flag=0;
	double **c = (double**)calloc(num_of_clusters, sizeof(double*));
	for(i=0; i<num_of_clusters; i++) c[i] = (double*)calloc(dimension, sizeof(double));

	normVector = (double *) malloc(sizeof(double)*num_points);
	distance   = (double *) malloc(sizeof(double)*num_points);

	for(i=0; i<num_points; i++){
		normVector[i]=norm(*(data_points+i),dimension);
	}

	//First Step -- determine first centroid
	for(i=0; i<num_points; i++){
		if(normVector[i]>max){ max=normVector[i];max_index = i;}
	}
	for(p=0;p<dimension;p++) c[num_of_centroids][p] = data_points[max_index][p];  
	num_of_centroids++;


	//Second Step -- determine the second centroid
	for (i = 0; i < num_points; i++) {
		distance[i] = calc_distance(data_points[i],c[0],dimension); 
	}

	max_distance = DBL_MIN;
	for(i=0; i<num_points; i++){
		if(distance[i]>max_distance){ max_distance=distance[i];max_index = i;}
	}
	for(p=0;p<dimension;p++) c[num_of_centroids][p] = data_points[max_index][p];  
	num_of_centroids++;



	//Third Step --  determine all remaining centroids 
	if(num_of_clusters>2){
		do{
			for (i=0; i<num_points; i++) {
				distance[i]=0;
				for(k=0; k<num_of_centroids; k++){

					if (check_match(data_points[i],c[k],dimension) == 1) {flag = 1; continue;}
					dist= calc_distance(data_points[i],c[k],dimension);

					if(dist <min_distance) min_distance = dist;
				}
				if(flag==1) {distance[i]=DBL_MIN;flag = 0;}
				else distance[i] = min_distance;
			}


			max_distance = DBL_MIN;
			for(i=0; i<num_points; i++){
				if(distance[i]>max_distance){ max_distance=distance[i];max_index = i;}
			}

			for(p=0;p<dimension;p++) c[num_of_centroids][p] = data_points[max_index][p];  
			num_of_centroids++;

		}while(num_of_centroids<num_of_clusters);
	}
	return c;

}

double  **random_initialization(double **data_points,int num_points,int dimension,int num_of_clusters){
	int i;
	double **c = (double**)calloc(num_of_clusters, sizeof(double*));
	for(i=0; i<num_of_clusters; i++) c[i] = (double*)calloc(dimension, sizeof(double));

	for(i=0;i<num_of_clusters;i++){
		memcpy(c[i],data_points[i],sizeof(double)*dimension);
	}

	return c;
}

int k_means(double **data_points,int *counts, double **c1,double **c,int *labels,int num_points, int dimension, int num_of_clusters){
	int h, i, j;
	int changes_flag;

	changes_flag=0;

	/* clear old counts and temp centroids */
	for (i = 0; i < num_of_clusters; counts[i++] = 0) {
		memset(c1[i],0,sizeof(double)*dimension);
	}

#pragma omp parallel for private(i,j) 
	for (h = 0; h < num_points; h++) {
		/* identify the closest cluster */
		double min_distance = DBL_MAX;
		int min_index;
		for (i = 0; i < num_of_clusters; i++) {
			double distance = calc_distance_et(data_points[h],c[i],dimension,min_distance);
			if (distance < min_distance) {
				min_index = i;                        /* vector h has centroid = i */
				min_distance = distance;

			}
		}

			
		/* update size and temp centroid of the destination cluster */
		if(min_index != labels[h]){
			#pragma omp atomic 
			changes_flag++;
			labels[h] = min_index;
		}

		for(j=dimension-1; j>=0; j--) 
			#pragma omp atomic
				c1[labels[h]][j] += data_points[h][j];

		#pragma omp atomic
			counts[labels[h]]++;
			
	}

	#pragma omp parallel for private(j)
	for (i = 0; i < num_of_clusters; i++) {
		if(counts[i]<=1){
			memcpy(c[i],c1[i],sizeof(double)*dimension);
		}else{
			for (j = 0; j < dimension; j++) {
				c[i][j] = c1[i][j]/counts[i];
			}
		}
	}

	return changes_flag;
}

int main(int argc,char **argv){
	int *labels;
	int *counts;
	double **c;
	double **c1;
	double **data_points=NULL;  
	int i,j;
	int changes=1;
	int num_points,dimension,num_of_clusters;
	clock_t start,stop;

	num_points=atoi(argv[1]);
	dimension=atoi(argv[2]);
	num_of_clusters=atoi(argv[3]);
	
	omp_set_num_threads(MAX_THREADS);

	if(num_points<num_of_clusters) {
		printf("This is not possible!!!\n");
		return -1;
	}

	printf("Start k-means with %d threads\n",omp_get_max_threads());

	labels = (int*)calloc(num_points, sizeof(int));
	counts = (int*)calloc(num_of_clusters, sizeof(int)); /* size of each cluster */

	data_points = (double **)malloc(num_points*sizeof(double*));
	for(i=0;i<num_points;i++)
		data_points[i]=(double*)malloc(dimension*sizeof(double));

	c1 = (double **)malloc(num_of_clusters*sizeof(double *)); /* temp centroids */
	for (i = 0; i < num_of_clusters; i++) {
		c1[i] = (double*)malloc(dimension*sizeof(double));
	}


	srand(1);
	for(i=0;i<num_points;i++){
		for(j=0;j<dimension;j++){
			data_points[i][j] = (double)(rand()%1024)/rand();
		}
	}

	// print_data(data_points,num_points,dimension);

	printf("K-means initialization...");
	//c = kkz_initialization(data_points,num_points,dimension,num_of_clusters);
	c = random_initialization(data_points,num_points,dimension,num_of_clusters);

	printf("succesfully completed\n");

	for(i=0;i<1000 && changes>0;i++){
		start = clock();
		changes = k_means(data_points,counts,c1,c,labels,num_points, dimension,num_of_clusters);
		stop = clock();

		printf("Iter: %4d, Diff: %6d, Duration: %6.3lf\n",i,changes,((double)stop-start)/(CLOCKS_PER_SEC*omp_get_max_threads()));
	}

	//free all dynamic memory
	free(labels);
	for (i = 0; i < num_of_clusters; i++) {
		free(c[i]);
		free(c1[i]);
	}
	free(c);
	free(c1);
	free(counts);


	return 1;
}