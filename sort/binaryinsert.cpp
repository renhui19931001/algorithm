#include <iostream>
#include <vector>
using namespace std;
void binary_insert(int*,int);
int main(){
	int a[] = {5,6,7,1,23,5,9,4};
	int length = sizeof(a)/sizeof(int);
	binary_insert(a,length);
	for(int i =0;i<length;i++)
		cout << a[i] << endl;
	return 0;
}
void binary_insert(int a[],int n){
	for(int i =1;i<n;i++){
		int left = 0;
		int right = i-1;
		int zhi = a[i];
		while(left <= right){
			int mid = (left+right)/2;
			if(a[mid] > zhi)
				right = mid-1;
			else
				left = mid+1;
		}
		for(int j =i-1;j >=left;j--)
			a[j+1] = a[j];
		a[left] = zhi;
	}
}