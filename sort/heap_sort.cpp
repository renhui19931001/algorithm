#include <iostream>
#include <vector>
using namespace std;
void build_min_tree(int*,int,int);
void heap_sort(int*,int);
int main(){
	int a[] = {5,6,7,1,23,5,9,4};
	int length = sizeof(a)/sizeof(int);
	heap_sort(a,length);
	for(int i =0;i<length;i++)
		cout << a[i] << endl;
	return 0;
}
void heap_sort(int a[],int n){
	int temp[100];
	for(int i =1;i<=n;i++)
		temp[i] = a[i-1];
	for(int i =n;i>=1;i--) build_min_tree(temp,n,i);
	for(int i =n;i >= 1;i--){
		a[n-i] = temp[1];
		temp[1] = temp[i];
		//here jizhu tiaojian
		build_min_tree(temp,i-1,1); 
	}
}
void build_min_tree(int temp[],int ans,int k){
	while(2*k+1 <= ans){
		if(temp[k] < temp[2*k]&&temp[k] < temp[2*k+1]) return;
		else if(temp[2*k] <temp[2*k+1]){
			swap(temp[k],temp[2*k]);
			k = 2*k;
		}else{
			swap(temp[k],temp[2*k + 1]);
			k = 2*k+1;
		}
	}
	if(2*k == ans && temp[2*k] < temp[k])
		swap(temp[k],temp[2*k]);
}