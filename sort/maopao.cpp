#include <iostream>
#include <vector>
using namespace std;
void maopao(int*,int);
int main(){
	int a[] = {5,6,7,1,23,5,9,4};
	int length = sizeof(a)/sizeof(int);
	maopao(a,length);
	for(int i =0;i<length;i++)
		cout << a[i] << endl;
	return 0;
}
void maopao(int a[],int n){
	for(int i =0;i<n;i++)
		for(int j = i+1;j<n;j++)
			if(a[j] < a[i])
				swap(a[i],a[j]);
}