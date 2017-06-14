#include <iostream>
#include <vector>
using namespace std;
void insert(int*,int);
int main(){
	int a[] = {5,6,7,1,23,5,9,4};
	int length = sizeof(a)/sizeof(int);
	insert(a,length);
	for(int i =0;i<length;i++)
		cout << a[i] << endl;
	return 0;
}
void insert(int a[],int n){
	for(int i =1;i<n;i++){
		int temp = a[i];
		int j =i-1;
		while(j>=0 && a[j] > temp){
			a[j+1] = a[j];
			j--;
		}
		a[j+1] = temp;
	}
}