#include <iostream>
#include <vector>
using namespace std;
void xuanze(int*,int);
int main(){
	int a[] = {5,6,7,1,23,5,9,4};
	int length = sizeof(a)/sizeof(int);
	xuanze(a,length);
	for(int i =0;i<length;i++)
		cout << a[i] << endl;
	return 0;
}
void xuanze(int a[],int n){
	for(int i =0;i<n;i++){
		int minNum = i;
		for(int j = i+1;j<n;j++){
			if(a[j]<a[minNum])
				minNum = j;
		} 
		swap(a[i],a[minNum]);
	}
}