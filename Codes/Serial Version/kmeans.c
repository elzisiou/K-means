/*
 *  kmeans.c 
 * serial version 2
 * Asimina Vouronikoy - Eleni Zisiou
 * 05/04/2013
 * compile : gcc -Wall kmeans.c -o kmeans -lm
 * run :  ./kmeans
 * Update 1 :This version uses kkz algorithm to pick initial centroids
 * Update 2 :This version stops the k-means iterations when no movement happens
 * Update 3: In this version k-means is called iteratively from main and the initialization passes as an argument
 */

#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>


int *labels;
int *counts;
double **c;
double **c1; 

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
  double distance =0;
  int j;
  for (j=0; j<dimension; j++){
    distance += pow(data_points[j] - c[j], 2);
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
  
  double normVector[num_points],distance[num_points];
  double max=DBL_MIN, max_distance = DBL_MIN , min_distance = DBL_MAX,dist;
  int i,j,k,p,max_index,num_of_centroids=0,flag=0;
  double **c = (double**)calloc(num_of_clusters, sizeof(double*));
  for(i=0; i<num_of_clusters; i++) c[i] = (double*)calloc(dimension, sizeof(double));
  
  for(i=0; i<num_points; i++){
    normVector[i]=norm(*(data_points+i),dimension);
  }
  
  //First Step -- determine first centroid
  for(i=0; i<num_points; i++){
    if(normVector[i]>max){ max=normVector[i];max_index = i;}
  }
  for(p=0;p<dimension;p++) c[num_of_centroids][p] = data_points[max_index][p];  
  printf("The %d centroid is  [%.3lf,%.3lf]\n",num_of_centroids+1, c[num_of_centroids][0], c[num_of_centroids][1]);
  num_of_centroids++;
  
  
  //Second Step -- determine the second centroid
  for (i = 0; i < num_points; i++) {
    distance[i] = 0;
    for (j = 0;j<dimension;j++){
      distance[i] += pow(data_points[i][j] - c[0][j], 2);
    }   
  }
  max_distance = DBL_MIN;
  for(i=0; i<num_points; i++){
    if(distance[i]>max_distance){ max_distance=distance[i];max_index = i;}
  }
  for(p=0;p<dimension;p++) c[num_of_centroids][p] = data_points[max_index][p];  
  printf("The %d centroid is  [%.3lf,%.3lf]\n",num_of_centroids+1, c[num_of_centroids][0], c[num_of_centroids][1]);
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
      printf("The %d centroid is  [%.3lf,%.3lf]\n",num_of_centroids+1, c[num_of_centroids][0], c[num_of_centroids][1]);
      num_of_centroids++;
      
    }while(num_of_centroids<num_of_clusters);
  }
  return c;
  
}


int k_means(double **data_points, int num_points, int dimension, int num_of_clusters){
  
  
  int h, i, j;
  int changes_flag;
  
  
  changes_flag=0;
  
  
  /* clear old counts and temp centroids */
  for (i = 0; i < num_of_clusters; counts[i++] = 0) {
    for (j = 0; j < dimension; j++)
      c1[i][j] = 0;
  }
  
  for (h = 0; h < num_points; h++) {
    /* identify the closest cluster */
    double min_distance = DBL_MAX;
    for (i = 0; i < num_of_clusters; i++) {
      double distance = 0;
      
      for (j = dimension; j > 0;j--)
	distance += pow(data_points[h][j] - c[i][j], 2);
      
      if (distance < min_distance) {
	labels[h] = i;                        /* vector h has centroid = i */
	min_distance = distance;
	
      }
    }
    /* update size and temp centroid of the destination cluster */
    
    for(j=dimension-1; j>0; j--) 
      c1[labels[h]][j] += data_points[h][j];
    
    counts[labels[h]]++;
  }
  /* update all centroids */
  for (i = 0; i < num_of_clusters; i++) { 
    for (j = 0; j < dimension; j++) {
      
      if(counts[i]>1)	{		/* find the mean !!*/
	if( c[i][j] != c1[i][j]/counts[i]){ changes_flag++; }
	c[i][j] = c1[i][j]/counts[i];
    }else{
      if( c[i][j] != c1[i][j]){ changes_flag++; }
      c[i][j] = c1[i][j];
    }
  }
}


return changes_flag;
}


int main(int argc,char **argv){
  
  double **data_points=NULL;  
  double **centroids=0;
  int i,j,h;
  int changes=1;
  int num_points,dimension,num_of_clusters;
  num_points=atoi(argv[1]);
  dimension=atoi(argv[2]);
  num_of_clusters=atoi(argv[3]);
  /* output cluster label for each data point */
  labels = (int*)calloc(num_points, sizeof(int));
  counts = (int*)calloc(num_of_clusters, sizeof(int)); /* size of each cluster */
  
  
  if(num_points<num_of_clusters) {
    printf("This is not possible!!!\n");
    return -1;
  }
  
  
  data_points = (double **)malloc(num_points*sizeof(double*));
  for(i=0;i<num_points;i++)
    data_points[i]=(double*)malloc(dimension*sizeof(double));
    srand(1);
  for(i=0;i<num_points;i++){
    for(j=0;j<dimension;j++){
      data_points[i][j] = rand()%10;
    }
  }
  
  // print_data(data_points,num_points,dimension);
  
  centroids = kkz_initialization(data_points,num_points,dimension,num_of_clusters);
  
  c = (double**)calloc(num_of_clusters, sizeof(double*));;
  c1 = (double**)calloc(num_of_clusters, sizeof(double*)); /* temp centroids */
  
  
  // initialization  
  for (h = i = 0; i < num_of_clusters; h += num_points/num_of_clusters, i++) {
    c1[i] = (double*)calloc(dimension, sizeof(double));
    c[i] = (double*)calloc(dimension, sizeof(double));
    
    for (j = dimension-1; j > 0;j--) c[i][j] = centroids[i][j];
  }
  
  for(i=0;changes>0;i++)
    changes = k_means(data_points, num_points, dimension,num_of_clusters);
  printf("epanalipseis %d\n",i);
  
  
  for (i = 0; i < num_points; i++) {
    printf("data point %d is in cluster %d\n", i, labels[i]);
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