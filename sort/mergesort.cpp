#include <iostream>
#include <vector>
using namespace std;
void merge(int*,int,int);
void merge_sort(int*,int,int,int);
int main(){
	int a[] = {5,6,7,1,23,5,9,4};
	int length = sizeof(a)/sizeof(int);
	merge(a,0,length-1);
	for(int i =0;i<length;i++)
		cout << a[i] << endl;
	return 0;
}
void merge(int a[],int left,int right){
	if(left <right){
		int mid = (left + right)/2;
		merge(a,left,mid);
		merge(a,mid+1,right);
		merge_sort(a,left,mid,right);
	}
}
void merge_sort(int a[],int left,int mid,int right){
	std::vector<int> num;
	int i = left;
	int j = mid+1;
	while(i<=mid && j <=right){
		if(a[i] < a[j])
			num.push_back(a[i++]);
		else
			num.push_back(a[j++]);
	}
	while(i<=mid)
		num.push_back(a[i++]);
	while(j <= right)
		num.push_back(a[j++]);
	for(int i =0;i<num.size();i++)
		a[left+i] = num[i];
}