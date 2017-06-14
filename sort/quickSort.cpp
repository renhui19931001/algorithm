#include <iostream>
#include <vector>
using namespace std;
void quicksort(int*,int,int);
int main(){
	int a[] = {5,6,7,1,23,5,9,4};
	int length = sizeof(a)/sizeof(int);
	quicksort(a,0,length-1);
	for(int i =0;i<length;i++)
		cout << a[i] << endl;
	return 0;
}
void quicksort(int a[],int left,int right){
	if(left < right){
		int x =a[left];
		int i = left;
		int j = right;
		while(i<j){
			while(a[j]>x && i<j)
				j--;
			if(i < j)
				a[i++] = a[j];
			while(a[i]<x && i<j)
				i++;
			if(i < j)
				a[j--] = a[i];
		}
		a[i] = x;
		quicksort(a,left,i-1);
		quicksort(a,i+1,right);
	}
}